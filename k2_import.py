# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software Foundation,
#  Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# ##### END GPL LICENSE BLOCK #####
from mathutils import Vector, Matrix

# determines the verbosity of loggin.
#   0 - no logging (fatal errors are still printed)
#   1 - standard logging
#   2 - verbose logging
#   3 - debug level. really boring (stuff like vertex data and verbatim lines)
IMPORT_LOG_LEVEL = 3

def log(msg):
    if IMPORT_LOG_LEVEL >= 1: print (msg)

def vlog(msg):
    if IMPORT_LOG_LEVEL >= 2: print (msg)

def dlog(msg):
    if IMPORT_LOG_LEVEL >= 3: print (msg)

def err(msg):
    log(msg)

import bpy
import struct,chunk

from bpy.props import *
def read_int(honchunk):
    return struct.unpack("<i",honchunk.read(4))[0]
def read_float(honchunk):
    return struct.unpack("<f",honchunk.read(4))[0]

def parse_links(honchunk,bone_names):
    mesh_index = read_int(honchunk)
    numverts = read_int(honchunk)
    log("links")
    vlog("mesh index: %d" % mesh_index)
    vlog("vertices number: %d" % numverts)
    vgroups = {}
    for i in range(numverts):
        num_weights = read_int(honchunk)
        weights = struct.unpack("<%df" % num_weights,honchunk.read(num_weights * 4))
        indexes = struct.unpack("<%dI" % num_weights,honchunk.read(num_weights * 4))
        for ii, index in enumerate(indexes):
            name = bone_names[index]
            if name not in vgroups:
                vgroups[name] = list()
            vgroups[name].append( (i,weights[ii] ) )
    honchunk.skip()
    return vgroups

def createTextureLayer(name, me, texFaces):
    uvtex = me.uv_textures.new()
    uvtex.name = name
    for n,tf in enumerate(texFaces):
        datum = uvtex.data[n]
        datum.uv1 = tf[0]
        datum.uv2 = tf[1]
        datum.uv3 = tf[2]
    return uvtex

def parse_vertices(honchunk):
    vlog('parsing vertices chunk')
    numverts = (honchunk.chunksize - 4)/12
    numverts = int(numverts)
    vlog('%d vertices' % numverts)
    meshindex = read_int(honchunk)
    return [struct.unpack("<3f", honchunk.read(12)) for i in range(int(numverts))]
def parse_sign(honchunk):
    vlog('parsing sign chunk')
    numverts = (honchunk.chunksize - 8)
    meshindex = read_int(honchunk)
    vlog(read_int(honchunk)) # huh?
    return [struct.unpack("<b", honchunk.read(1)) for i in range(numverts)]
def parse_faces(honchunk,version):
    vlog('parsing faces chunk')
    meshindex = read_int(honchunk)
    numfaces = read_int(honchunk)
    vlog('%d faces' % numfaces)
    if version == 3:
        size = struct.unpack('B',honchunk.read(1))[0]
    elif version == 1:
        size = 4
    if size == 2:
        return [struct.unpack("<3H", honchunk.read(6)) for i in range(numfaces)]
    elif size ==1:
        return [struct.unpack("<3B", honchunk.read(3)) for i in range(numfaces)]
    elif size == 4:
        return [struct.unpack("<3I", honchunk.read(12)) for i in range(numfaces)]
    else:
        log("unknown size for faces:%d" % size)
        return []
def parse_normals(honchunk):
    vlog('parsing normals chunk')
    numverts = (honchunk.chunksize - 4)/12
    numverts = int(numverts)
    vlog('%d normals' % numverts)
    meshindex = read_int(honchunk)
    return [struct.unpack("<3f", honchunk.read(12)) for i in range(numverts)]
def parse_texc(honchunk,version):
    vlog('parsing uv texc chunk')
    numverts = int((honchunk.chunksize - 4)/8)
    numverts = int(numverts)
    vlog('%d texc' % numverts)
    meshindex = read_int(honchunk)
    if version == 3:
        vlog(read_int(honchunk)) # huh?
    return [struct.unpack("<2f", honchunk.read(8)) for i in range(numverts)]   
def parse_colr(honchunk):
    vlog('parsing vertex colours chunk')
    numverts = (honchunk.chunksize - 4)/4
    numverts = int(numverts)
    meshindex = read_int(honchunk)
    return [struct.unpack("<4B", honchunk.read(4)) for i in range(numverts)]

def roundVector(vec,dec=17):
    fvec=[]
    for v in vec:
        fvec.append(round(v,dec))
    return Vector(fvec)

def roundMatrix(mat,dec=17):
    fmat = []
    for row in mat:
        fmat.append(roundVector(row,dec))
    return Matrix(fmat)    

def CreateBlenderMesh(filename, objname):
    file = open(filename,'rb')
    if not file:
        log("can't open file")
        return
    sig = file.read(4)
    if sig != b'SMDL':
        err('unknown file signature')
        return

    try:
        honchunk = chunk.Chunk(file,bigendian=0,align=0)
    except EOFError:
        log('error reading first chunk')
        return
    if honchunk.getname() != b'head':
        log('file does not start with head chunk!')
        return
    version = read_int(honchunk)
    num_meshes = read_int(honchunk)
    num_sprites = read_int(honchunk)
    num_surfs = read_int(honchunk)
    num_bones = read_int(honchunk)

    vlog("Version %d" % version)
    vlog("%d mesh(es)" % num_meshes)
    vlog("%d sprites(es)" % num_sprites)
    vlog("%d surfs(es)" % num_surfs)
    vlog("%d bones(es)" % num_bones)
    vlog("bounding box: (%f,%f,%f) - (%f,%f,%f)" % \
        struct.unpack("<ffffff", honchunk.read(24)))
    honchunk.skip()

    scn= bpy.context.scene

    try:
        honchunk = chunk.Chunk(file,bigendian=0,align=0)
    except EOFError:
        log('error reading bone chunk')
        return


    #read bones

    #create armature object
    armature_data = bpy.data.armatures.new('%s_Armature' % objname)
    armature_data.show_names = True
    rig = bpy.data.objects.new('%s_Rig' % objname, armature_data)
    scn.objects.link(rig)
    scn.objects.active = rig
    rig.select = True

    # armature = armature_object.getData()
    # if armature is None:
    # base_name = Blender.sys.basename(file_object.name)
    # armature_name = Blender.sys.splitext(base_name)[0]
    # armature = Blender.Armature.New(armature_name)
    # armature_object.link(armature)
    #armature.drawType = Blender.Armature.STICK
    #armature.envelopes = False
    #armature.vertexGroups = True
       
    #bpy.ops.object.editmode_toggle()
    bpy.ops.object.mode_set(mode='EDIT')

    bones = []
    bone_names = []
    parents = []
    for i in range(num_bones):
        name = ''
        parent_bone_index = read_int(honchunk)

        if version == 3:
            inv_matrix = Matrix((struct.unpack('<3f', honchunk.read(12)) + (0.0,),
                         struct.unpack('<3f', honchunk.read(12)) + (0.0,),
                         struct.unpack('<3f', honchunk.read(12)) + (0.0,),
                         struct.unpack('<3f', honchunk.read(12)) + (1.0,)))

            matrix = Matrix((struct.unpack('<3f', honchunk.read(12)) + (0.0,),
                     struct.unpack('<3f', honchunk.read(12)) + (0.0,),
                     struct.unpack('<3f', honchunk.read(12)) + (0.0,),
                     struct.unpack('<3f', honchunk.read(12)) + (1.0,)))

            name_length = struct.unpack("B" , honchunk.read(1))[0]
            name = honchunk.read(name_length)

            honchunk.read(1) #zero
        elif version == 1:
            name = ''
            pos = honchunk.tell() - 4
            b = honchunk.read(1)
            while b != '\0':
                name += b
                b = honchunk.read(1)
            honchunk.seek(pos + 0x24)
            inv_matrix = Matrix((struct.unpack('<4f', honchunk.read(16)),
                         struct.unpack('<4f', honchunk.read(16)),
                         struct.unpack('<4f', honchunk.read(16)),
                         struct.unpack('<4f', honchunk.read(16))))

            matrix = Matrix((struct.unpack('<4f', honchunk.read(16)),
                     struct.unpack('<4f', honchunk.read(16)),
                     struct.unpack('<4f', honchunk.read(16)),
                     struct.unpack('<4f', honchunk.read(16))))

        name = name.decode()
        log("bone name: %s,parent %d" % (name,parent_bone_index))
        bone_names.append(name)
        edit_bone = armature_data.edit_bones.new(name)
        edit_bone.use_local_location = True
        edit_bone.use_inherit_rotation = True
        matrix = roundMatrix(matrix,4)
        #(trans, rot, scale) = matrix.decompose()
        edit_bone.head = matrix[3].xyz
        #edit_bone.tail = edit_bone.head + trans
        #edit_bone.roll = rot.to_euler().z
        edit_bone.length = 1
        #edit_bone.transform(roundMatrix(matrix,4))
        #edit_bone.use_connect=False
        #print (edit_bone.head)
        #print (edit_bone.tail)
        #print (edit_bone.roll)
        #print (edit_bone.matrix)
        #print (edit_bone.length)
        parents.append(parent_bone_index)
        bones.append(edit_bone)
    for i in range(num_bones):
        if parents[i] != -1:
            bones[i].parent = bones[parents[i]]

    honchunk.skip()

    bpy.ops.object.mode_set(mode='OBJECT')
    rig.show_x_ray = True
    rig.update_tag()
    scn.update()

    try:
        honchunk = chunk.Chunk(file,bigendian=0,align=0)
    except EOFError:
        log('error reading mesh chunk')
        return
    while honchunk.getname() == b'mesh':
        verts = []
        faces = []
        signs = []
        nrml = []
        texc = []
        colors = []
        #read mesh chunk
        vlog("mesh index: %d" % read_int(honchunk))
        mode = 1
        if version == 3:
            mode = read_int(honchunk)
            vlog("mode: %d" % mode)
            vlog("vertices count: %d" % read_int(honchunk))
            vlog("bounding box: (%f,%f,%f) - (%f,%f,%f)" % \
                struct.unpack("<ffffff", honchunk.read(24)))
            bone_link = read_int(honchunk)
            vlog("bone link: %d" % bone_link)
            sizename = struct.unpack('B',honchunk.read(1))[0]
            sizemat = struct.unpack('B',honchunk.read(1))[0]
            meshname = honchunk.read(sizename)
            honchunk.read(1) # zero
            materialname = honchunk.read(sizemat)
        elif version == 1:
            bone_link = -1
            pos = honchunk.tell() - 4
            b = honchunk.read(1)
            meshname = ''
            while b != '\0':
                meshname += b
                b = honchunk.read(1)
            honchunk.seek(pos + 0x24)

            b = honchunk.read(1)
            materialname = ''
            while b != '\0':
                materialname += b
                b = honchunk.read(1)

        honchunk.skip()

        meshname = meshname.decode()
        materialname = materialname.decode()

        if mode == 1 or not False:#SKIP_NON_PHYSIQUE_MESHES:
            msh = bpy.data.meshes.new(meshname)

            #msh = bpy.data.meshes.new(meshname)
            #msh.mode |= Blender.Mesh.Modes.AUTOSMOOTH
            #obj = scn.objects.new(msh)

        while 1:
            try:
                honchunk = chunk.Chunk(file,bigendian=0,align=0)
            except EOFError:
                vlog('error reading chunk')
                break
            if honchunk.getname == b'mesh':
                break
            elif mode != 1 and SKIP_NON_PHYSIQUE_MESHES:
                honchunk.skip()
            else:
                if honchunk.getname() == b'vrts':
                    verts = parse_vertices(honchunk)
                elif honchunk.getname() == b'face':
                    faces = parse_faces(honchunk,version)
                elif honchunk.getname() == b'nrml':
                    nrml = parse_normals(honchunk)
                elif honchunk.getname() == b'texc':
                    texc = parse_texc(honchunk,version)
                elif honchunk.getname() == b'colr':
                    colors = parse_colr(honchunk)
                elif honchunk.getname() == b'lnk1' or honchunk.getname() == b'lnk3':
                    vgroups = parse_links(honchunk,bone_names)
                elif honchunk.getname() == b'sign':
                    signs = parse_sign(honchunk)
                else:
                    vlog('unknown chunk: %s' % honchunk.chunkname)
                    honchunk.skip()
        if mode != 1 and False:#SKIP_NON_PHYSIQUE_MESHES:
            continue

           
        msh.materials.append(bpy.data.materials.new(materialname))

        msh.from_pydata( verts, [], faces )
        msh.update(calc_edges=True)

         
        if True:#flipuv:
            for t in range(len(texc)):
                texc[t] = ( texc[t][0], 1-texc[t][1] )


        # Generate texCoords for faces
        texcoords = []
        for face in faces:
            texcoords.append( [ texc[vert_id] for vert_id in face ] )
            
        uvMain = createTextureLayer("UVMain", msh, texcoords)
                        
        for vertex, normal in zip(msh.vertices, nrml):
            vertex.normal = normal
        


        obj = bpy.data.objects.new('%s_Object' % meshname, msh)
        # Link object to scene
        scn.objects.link(obj)
        scn.objects.active = obj
        scn.update()

        if bone_link >=0 :
            grp = obj.vertex_groups.new(bone_names[bone_link])
            grp.add(list(range(len(msh.vertices))),1.0,'REPLACE')


        for name in vgroups.keys():
            grp = obj.vertex_groups.new(name)
            for (v, w) in vgroups[name]:
                grp.add([v], w, 'REPLACE')
        
        mod = obj.modifiers.new('MyRigModif', 'ARMATURE')
        mod.object = rig
        mod.use_bone_envelopes = False
        mod.use_vertex_groups = True


        if False:#removedoubles:
            obj.select = True
            bpy.ops.object.mode_set(mode='EDIT', toggle=False)
            bpy.ops.mesh.remove_doubles()
            bpy.ops.object.mode_set(mode='OBJECT', toggle=False)
            obj.select = False


        bpy.context.scene.objects.active = rig
        rig.select = True
        bpy.ops.object.mode_set(mode='POSE', toggle=False)
        pose = rig.pose
        for b in pose.bones:
            b.rotation_mode = "QUATERNION"
        bpy.ops.object.mode_set(mode='OBJECT', toggle=False)
        rig.select = False
        bpy.context.scene.objects.active = None


    scn.update()
    return ( obj, rig )



#def addMeshObj(mesh, objName):
    #scn = bpy.context.scene

    #for o in scn.objects:
        #o.select = False

    #mesh.update()
    #mesh.validate()

    #nobj = bpy.data.objects.new(objName, mesh)
    #scn.objects.link(nobj)
    #nobj.select = True

    #if scn.objects.active is None or scn.objects.active.mode == 'OBJECT':
        #scn.objects.active = nobj

def CreateBlenderClip(filename,objName):
    dlog('Importing clip')

def read(filepath):
    #convert the filename to an object name
    objName = bpy.path.display_name_from_filepath(filepath)
    CreateBlenderMesh(filepath, objName)
    #addMeshObj(mesh, objName)

# package manages registering

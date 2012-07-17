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
from mathutils import Vector, Matrix, Euler
import math
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

#########################################
## "Visual Transform" helper functions ##
#########################################

def get_pose_matrix_in_other_space(mat, pose_bone):
    """ Returns the transform matrix relative to pose_bone's current
        transform space.  In other words, presuming that mat is in
        armature space, slapping the returned matrix onto pose_bone
        should give it the armature-space transforms of mat.
        TODO: try to handle cases with axis-scaled parents better.
    """
    rest = pose_bone.bone.matrix_local.copy()
    rest_inv = rest.inverted()
    if pose_bone.parent:
        par_mat = pose_bone.parent.matrix.copy()
        par_inv = pose_bone.parent.matrix.inverted()
        par_rest = pose_bone.parent.bone.matrix_local.copy()
    else:
        par_mat = Matrix()
        par_inv = Matrix()
        par_rest = Matrix()

    # Get matrix in bone's current transform space
    smat = rest_inv * (par_rest * (par_inv * mat))

    # Compensate for non-inherited rotation/scale
    if not pose_bone.bone.use_inherit_rotation:
        loc = mat.to_translation()
        loc -= (par_mat*(par_rest.inverted() * rest)).to_translation()
        loc *= rest.inverted().to_quaternion()
        if pose_bone.bone.use_inherit_scale:
            t = par_mat.to_scale()
            par_scale = Matrix().Scale(t[0], 4, Vector((1,0,0)))
            par_scale *= Matrix().Scale(t[1], 4, Vector((0,1,0)))
            par_scale *= Matrix().Scale(t[2], 4, Vector((0,0,1)))
        else:
            par_scale = Matrix()

        smat = rest_inv * mat * par_scale.inverted()
        smat[3][0] = loc[0]
        smat[3][1] = loc[1]
        smat[3][2] = loc[2]
    elif not pose_bone.bone.use_inherit_scale:
        loc = smat.to_translation()
        rot = smat.to_quaternion()
        scl = mat.to_scale()

        smat = Matrix().Scale(scl[0], 4, Vector((1,0,0)))
        smat *= Matrix().Scale(scl[1], 4, Vector((0,1,0)))
        smat *= Matrix().Scale(scl[2], 4, Vector((0,0,1)))
        smat *= Matrix.Rotation(rot.angle, 4, rot.axis)
        smat[3][0] = loc[0]
        smat[3][1] = loc[1]
        smat[3][2] = loc[2]

    # Compensate for non-local location
    if not pose_bone.bone.use_local_location:
        loc = smat.to_translation() * (par_rest.inverted() * rest).to_quaternion()
        smat[3][0] = loc[0]
        smat[3][1] = loc[1]
        smat[3][2] = loc[2]

    return smat


def get_local_pose_matrix(pose_bone):
    """ Returns the local transform matrix of the given pose bone.
    """
    return get_pose_matrix_in_other_space(pose_bone.matrix, pose_bone)


def set_pose_translation(pose_bone, mat):
    """ Sets the pose bone's translation to the same translation as the given matrix.
        Matrix should be given in bone's local space.
    """
    pose_bone.location = mat.to_translation()


def set_pose_rotation(pose_bone, mat):
    """ Sets the pose bone's rotation to the same rotation as the given matrix.
        Matrix should be given in bone's local space.
    """
    q = mat.to_quaternion()

    if pose_bone.rotation_mode == 'QUATERNION':
        pose_bone.rotation_quaternion = q
    elif pose_bone.rotation_mode == 'AXIS_ANGLE':
        pose_bone.rotation_axis_angle[0] = q.angle
        pose_bone.rotation_axis_angle[1] = q.axis[0]
        pose_bone.rotation_axis_angle[2] = q.axis[1]
        pose_bone.rotation_axis_angle[3] = q.axis[2]
    else:
        pose_bone.rotation_euler = q.to_euler(pose_bone.rotation_mode)


def set_pose_scale(pose_bone, mat):
    """ Sets the pose bone's scale to the same scale as the given matrix.
        Matrix should be given in bone's local space.
    """
    pose_bone.scale = mat.to_scale()


def match_pose_translation(pose_bone, target_bone):
    """ Matches pose_bone's visual translation to target_bone's visual
        translation.
        This function assumes you are in pose mode on the relevant armature.
    """
    mat = get_pose_matrix_in_other_space(target_bone.matrix, pose_bone)
    set_pose_translation(pose_bone, mat)
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.mode_set(mode='POSE')


def match_pose_rotation(pose_bone, target_bone):
    """ Matches pose_bone's visual rotation to target_bone's visual
        rotation.
        This function assumes you are in pose mode on the relevant armature.
    """
    mat = get_pose_matrix_in_other_space(target_bone.matrix, pose_bone)
    set_pose_rotation(pose_bone, mat)
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.mode_set(mode='POSE')


def match_pose_scale(pose_bone, target_bone):
    """ Matches pose_bone's visual scale to target_bone's visual
        scale.
        This function assumes you are in pose mode on the relevant armature.
    """
    mat = get_pose_matrix_in_other_space(target_bone.matrix, pose_bone)
    set_pose_scale(pose_bone, mat)
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.mode_set(mode='POSE')




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
        if num_weights > 0:
            weights = struct.unpack("<%df" % num_weights,honchunk.read(num_weights * 4))
            indexes = struct.unpack("<%dI" % num_weights,honchunk.read(num_weights * 4))
        else:
            weights = indexes = []
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
def vec_roll_to_mat3(vec, roll):
    target = Vector((0,1,0))
    nor = vec.normalized()
    axis = target.cross(nor)
    if axis.dot(axis) > 0.000001:
        axis.normalize()
        theta = target.angle(nor)
        bMatrix = Matrix.Rotation(theta, 3, axis)
    else:
        updown = 1 if target.dot(nor) > 0 else -1
        bMatrix = Matrix.Scale(updown, 3)
    rMatrix = Matrix.Rotation(roll, 3, nor)
    mat = rMatrix * bMatrix
    return mat

def mat3_to_vec_roll(mat):
    vec = mat.col[1]
    vecmat = vec_roll_to_mat3(mat.col[1], 0)
    vecmatinv = vecmat.inverted()
    rollmat = vecmatinv * mat
    roll = math.atan2(rollmat[0][2], rollmat[2][2])
    return vec, roll

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
        matrix.transpose()
        matrix = roundMatrix(matrix,4)
        pos = matrix.to_translation()
        #pos = Vector(matrix[3][:3])
        axis, roll = mat3_to_vec_roll(matrix.to_3x3())
        bone = armature_data.edit_bones.new(name)
        print(matrix)
        print(pos)
        bone.head = pos
        bone.tail = pos + axis
        bone.roll = roll
        print ('coool')
        parents.append(parent_bone_index)
        bones.append(bone)
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
            elif mode != 1 and False:#SKIP_NON_PHYSIQUE_MESHES:
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
############################## 
#CLIPS
##############################
    

MKEY_X,MKEY_Y,MKEY_Z,\
MKEY_PITCH,MKEY_ROLL,MKEY_YAW,\
MKEY_VISIBILITY,\
MKEY_SCALE_X,MKEY_SCALE_Y,MKEY_SCALE_Z \
    = range(10)

def bone_depth(bone):
    if not bone.parent:
        return 0
    else:
        return 1+bone_depth(bone.parent)
        
def getTransformMatrix(motions,bone,i,version):
    motion = motions[bone.name]
    #translation
    if i >= len(motion[MKEY_X]):
        x = motion[MKEY_X][-1]
    else:
        x = motion[MKEY_X][i]
    
    if i >= len(motion[MKEY_Y]):
        y = motion[MKEY_Y][-1]
    else:
        y = motion[MKEY_Y][i]
    
    if i >= len(motion[MKEY_Z]):
        z = motion[MKEY_Z][-1]
    else:
        z = motion[MKEY_Z][i]

    #rotation
    if i >= len(motion[MKEY_PITCH]):
        rx = motion[MKEY_PITCH][-1]
    else:
        rx = motion[MKEY_PITCH][i]
    
    if i >= len(motion[MKEY_ROLL]):
        ry = motion[MKEY_ROLL][-1]
    else:
        ry = motion[MKEY_ROLL][i]

    if i >= len(motion[MKEY_YAW]):
        rz = motion[MKEY_YAW][-1]
    else:
        rz = motion[MKEY_YAW][i]
    
    #scaling
    if version == 1:
        if i >= len(motion[MKEY_SCALE_X]):
            sx = motion[MKEY_SCALE_X][-1]
        else:
            sx = motion[MKEY_SCALE_X][i]
        sy = sz = sx
    else:
        if i >= len(motion[MKEY_SCALE_X]):
            sx = motion[MKEY_SCALE_X][-1]
        else:
            sx = motion[MKEY_SCALE_X][i]
        
        if i >= len(motion[MKEY_SCALE_Y]):
            sy = motion[MKEY_SCALE_Y][-1]
        else:
            sy = motion[MKEY_SCALE_Y][i]

        if i >= len(motion[MKEY_SCALE_Z]):
            sz = motion[MKEY_SCALE_Z][-1]
        else:
            sz = motion[MKEY_SCALE_Z][i]
    scale = Vector([sx,sy,sz])
    bone_rotation_matrix = Euler((math.radians(rx),math.radians(ry),math.radians(rz)),'YXZ').to_matrix().to_4x4()

    bone_rotation_matrix = Matrix.Translation(\
        Vector((x,y,z))) * bone_rotation_matrix

    return bone_rotation_matrix,scale
    
def AnimateBone(name,pose,motions,num_frames,armature,armOb,version):
    if name not in armature.bones.keys():
        log ('%s not found in armature' % name)
        return
    motion = motions[name]
    bone = armature.bones[name]
    bone_rest_matrix = Matrix(bone.matrix_local)
    print(name)
    print('rest',bone_rest_matrix)
    
    if bone.parent is not None:
            parent_bone = bone.parent
            parent_rest_bone_matrix = Matrix(parent_bone.matrix_local)
            print('parent ',parent_rest_bone_matrix)
            parent_rest_bone_matrix.invert()
            #bone_rest_matrix *= parent_rest_bone_matrix
            bone_rest_matrix = parent_rest_bone_matrix * bone_rest_matrix
    
    bone_rest_matrix_inv = Matrix(bone_rest_matrix)
    print('1rest inv with parent',bone_rest_matrix_inv)
    bone_rest_matrix_inv.invert()

    print('rest inv with parent',bone_rest_matrix_inv)

    pbone = pose.bones[name]
    #pbone.rotation_mode = 'XYZ'
    prev_euler = Euler()
    #num_frames = 1
    for i in range(0, num_frames):
        transform,size = getTransformMatrix(motions,bone,i,version)
        print ('transform')
        print (transform)
        #x = 1/0
        #transform = get_pose_matrix_in_other_space(transform,pbone)
        #transform *= bone_rest_matrix_inv
        transform = bone_rest_matrix_inv * transform
        print('final transform')
        print(transform)
        #pbone.matrix = bone_rest_matrix_inv * transform - bone_rest_matrix
        #pbone.matrix = get_pose_matrix_in_other_space(transform,pbone)
        #pbone.matrix = transform
        #pbone.matrix = bone_rest_matrix_inv * transform
        #pbone.matrix = bone_rest_matrix_inv * transform


        #x = 1/0

        #pbone.scale = scale

        #bone_rotation_matrix = bone_rest_matrix_inv * bone_rotation_matrix * bone_rest_matrix
        #euler = bone_rotation_matrix.to_euler('ZXY', prev_euler)
        #euler = bone_rotation_matrix.to_euler()
        #dlog(euler)
        #pbone.rotation_euler = euler
        pbone.rotation_quaternion = transform.to_quaternion()
        #pbone.rotation_euler = Euler((90.0,45.0,32.0))
        #prev_euler = euler

        #pbone.location =  (bone_rest_matrix_inv * bone_translation_matrix - bone_rest_matrix).to_translation()
        #pbone.location =  (bone_rest_matrix_inv * bone_translation_matrix).to_translation()
        #pbone.location =  (bone_translation_matrix - bone_rest_matrix).to_translation()
        #pbone.location =  (bone_translation_matrix * bone_rest_matrix_inv).to_translation()
        #pbone.location = transform.to_translation()
        #if i == 0:
        #print((name,i))
        #dlog(pbone.matrix_channel)
        #dlog(bone_translation_matrix)
        #dlog(pbone.location)
        #print('___________')
        pbone.location =  (transform).to_translation()

        #pbone.keyframe_insert(data_path='rotation_euler',frame=i)
        pbone.keyframe_insert(data_path='rotation_quaternion',frame=i+1)
        pbone.keyframe_insert(data_path='location',frame=i+1)
        #pbone.keyframe_insert(data_path='scale',frame=i)


def CreateBlenderClip(filename,clipname):
    file = open(filename,'rb')
    if not file:
        log("can't open file")
        return
    sig = file.read(4)
    if sig != b'CLIP':
        err('unknown file signature')
        return

    try:
        clipchunk = chunk.Chunk(file,bigendian=0,align=0)
    except EOFError:
        log('error reading first chunk')
        return
    version = read_int(clipchunk)
    num_bones = read_int(clipchunk)
    num_frames = read_int(clipchunk)
    vlog ("version: %d" % version)
    vlog ("num bones: %d" % num_bones)
    vlog ("num frames: %d" % num_frames)
    

    #objList = Blender.Object.GetSelected()
    #if len(objList) != 1:
        #err('select needed armature only')
    #armOb = objList[0]
    #action = Blender.Armature.NLA.NewAction(clipname)
    #action.setActive(armOb)
    #pose = armOb.getPose()
    #armature = armOb.getData()
    armOb = bpy.context.selected_objects[0]
    armOb.animation_data_create()
    armature = armOb.data
    action = bpy.data.actions.new(name=clipname)
    armOb.animation_data.action = action
    pose = armOb.pose
    
    bone_index = -1
    motions = {}
    

    while 1:
        try:
            clipchunk = chunk.Chunk(file,bigendian=0,align=0)
        except EOFError:
            break
        if version == 1:
            name = clipchunk.read(32)
            if '\0' in name:
                name = name[:name.index('\0')]
        boneindex = read_int(clipchunk)
        keytype = read_int(clipchunk)
        numkeys = read_int(clipchunk)
        if version > 1:
            namelength = struct.unpack("B",clipchunk.read(1))[0]
            name = clipchunk.read(namelength)
            clipchunk.read(1)
        name = name.decode("utf8")
            
        if name not in motions:
            motions[name] = {}
        dlog ("%s,boneindex: %d,keytype: %d,numkeys: %d" % \
            (name,boneindex,keytype,numkeys))
        if keytype == MKEY_VISIBILITY:
            data = struct.unpack("%dB"  % numkeys,clipchunk.read(numkeys))
        else:
            data = struct.unpack("<%df" % numkeys,clipchunk.read(numkeys * 4))
        motions[name][keytype] = list(data)
        clipchunk.skip()
    #file read, now animate that bastard!
    for bone_name in motions:
        AnimateBone(bone_name,pose,motions,num_frames,armature,armOb,version)
    #pose.update()

def readclip(filepath):
    objName = bpy.path.display_name_from_filepath(filepath)
    CreateBlenderClip(filepath, objName)

def read(filepath):
    objName = bpy.path.display_name_from_filepath(filepath)
    CreateBlenderMesh(filepath, objName)

# package manages registering

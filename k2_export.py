# Copyright (c) 2010 Anton Romanov
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
import bpy,bmesh
from io import BytesIO
import struct

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
def bone_depth(bone):
    if not bone.parent:
        return 0
    else:
        return 1+bone_depth(bone.parent)

def generate_bbox(meshes):
    # need to transform verts here?
    xx = []
    yy = []
    zz = []
    for mesh in meshes:
        nv = [v.co for v in mesh.verts]
        xx += [ co[0] for co in nv ]
        yy += [ co[1] for co in nv ]
        zz += [ co[2] for co in nv ]
    return [min(xx),min(yy),min(zz),max(xx),max(yy),max(zz)]
def create_mesh_data(mesh,vert,index,name,mname):
    meshdata = BytesIO()
    meshdata.write(struct.pack("<i",index))
    meshdata.write(struct.pack("<i",1)) # mode? huh? dunno...
    meshdata.write(struct.pack("<i",len(vert))) # vertices count
    meshdata.write(struct.pack("<6f",*generate_bbox([mesh]))) # bounding box
    meshdata.write(struct.pack("<i",-1)) # bone link... dunno... TODO
    meshdata.write(struct.pack("<B",len(name))) 
    meshdata.write(struct.pack("<B",len(mname))) 
    meshdata.write(name)
    meshdata.write(struct.pack("<B",0)) 
    meshdata.write(mname)
    meshdata.write(struct.pack("<B",0)) 
    return meshdata.getvalue()
    
def create_vrts_data(verts,meshindex):
    data = BytesIO()
    data.write(struct.pack("<i",meshindex))
    for v in verts:
        data.write(struct.pack("<3f",*v.co))
    return data.getvalue()
    
def create_face_data(verts,faces,meshindex):
    data = BytesIO()
    data.write(struct.pack("<i",meshindex))
    data.write(struct.pack("<i",len(faces)))
    
    if len(verts) < 255 :
        data.write(struct.pack("<B",1))
        str = '<3B'
    else:
        data.write(struct.pack("<B",2))
        str = '<3H'
    for f in faces:
        data.write(struct.pack(str,*f))
    return data.getvalue()

def create_tang_data(tang,meshindex):
    data = BytesIO()
    data.write(struct.pack("<i",meshindex))
    data.write(struct.pack("<i",0)) # huh?
    for t in tang:
        data.write(struct.pack('<3f',*list(t)))
    return data.getvalue()

def write_block(file,name,data):
    file.write(name.encode('utf8')[:4])
    file.write(struct.pack("<i",len(data)))
    file.write(data)

def create_texc_data(texc,meshindex):
    #if flip_uv:
    for i in range(len(texc)):
        texc[i] = [texc[i][0],1.0-texc[i][1]]
    data = BytesIO()
    data.write(struct.pack("<i",meshindex))
    data.write(struct.pack("<i",0)) # huh?
    for t in texc:
        data.write(struct.pack("<2f",*t))
    return data.getvalue()

def create_colr_data(colr,meshindex):
    data = BytesIO()
    data.write(struct.pack("<i",meshindex))
    for c in colr:
        data.write(struct.pack("<4B",c.r,c.g,c.b,c.a))
    return data.getvalue()

def create_nrml_data(verts,meshindex):
    data = BytesIO()
    data.write(struct.pack("<i",meshindex))
    for v in verts:
        data.write(struct.pack("<3f",*v.normal))
    return data.getvalue()

def create_lnk1_data(lnk1,meshindex,bone_indices):
    data = BytesIO()
    data.write(struct.pack("<i",meshindex))
    data.write(struct.pack("<i",len(lnk1)))
    for influences in lnk1:
        influences = [inf for inf in influences if inf[0] in bone_indices]
        l = len(influences)
        data.write(struct.pack("<i",l))
        if l > 0:
            data.write(struct.pack('<%df' % l,\
                *[inf[1] for inf in influences]))
            data.write(struct.pack('<%dI' % l,\
                *[bone_indices[inf[0]] for inf in influences]))
    return data.getvalue()

def create_sign_data(meshindex,sign):
    data = BytesIO()
    data.write(struct.pack("<i",meshindex))
    data.write(struct.pack("<i",0))
    for s in sign:
        data.write(struct.pack("<b",s))
    return data.getvalue()

def calcFaceSigns(ftexc):
    fsigns = []
    for uv in ftexc:
        if ((uv[1][0] - uv[0][0]) * (uv[2][1] - uv[1][1]) - (uv[1][1] - uv[0][1]) * (uv[2][0] - uv[1][0])) > 0:
            fsigns.append((0,0,0))
        else:
            fsigns.append((-1,-1,-1))
    return fsigns

def face_to_vertices_dup(faces,fdata,verts):
    vdata = [None]*len(verts)
    for fi,f in enumerate(faces):
        for vi,v in enumerate(f):
            if vdata[v] is None or vdata[v] == fdata[fi][vi]:
                vdata[v] = fdata[fi][vi]
            else:
                newind = len(verts)
                verts.append(verts[v])
                faces[fi][vi] = newind
                vdata.append(fdata[fi][vi])
    return vdata

def face_to_vertices(faces,fdata,verts):
    vdata = [None]*len(verts)
    for fi,f in enumerate(faces):
        for vi,v in enumerate(f):
            vdata[v] = fdata[fi][vi]
    return vdata

def create_bone_data(armature,armMatrix,transform):
    bones = []
    for bone in sorted(armature.bones.values(),key=bone_depth):
        bones.append(bone.name)
    bonedata = BytesIO()
    
    for name in bones:
        bone = armature.bones[name]
        base = bone.matrix_local.copy()
        if transform:
            base *= armMatrix
        baseInv = base.copy()
        baseInv.invert()
        if bone.parent :
            parent_index = bones.index(bone.parent.name)
        else:
            parent_index = -1
        baseInv.transpose()
        base.transpose()
        #parent bone index
        bonedata.write(struct.pack("<i",parent_index))
        #inverted matrix
        bonedata.write(struct.pack('<12f', *sum([list(row[0:3]) for row in baseInv],[])))
        #base matrix
        bonedata.write(struct.pack('<12f', *sum([list(row[0:3]) for row in base],[])))
        #bone name
        name = name.encode('utf8')
        bonedata.write(struct.pack("B" ,len(name)))
        bonedata.write(name)
        bonedata.write(struct.pack("B" ,0))
    return bones,bonedata.getvalue()

def export_k2_mesh(filename, applyMods):
    meshes = []
    armature = None
    for obj in bpy.context.selected_objects:
        if obj.type == 'MESH':
            matrix = obj.matrix_world

            if (applyMods):
                me = obj.to_mesh(bpy.context.scene, True, "PREVIEW")
            else:
                me = obj.data
            bm = bmesh.new()
            bm.from_mesh(me)
            me = bm
            me.transform(matrix)

            meshes.append((obj,me))
        elif obj.type == 'ARMATURE':
            armature = obj.data
            armMatrix = obj.matrix_world

    if armature:
        armature.pose_position = 'REST'
        bone_indices,bonedata = create_bone_data(armature,armMatrix,applyMods)

    headdata = BytesIO()
    headdata.write(struct.pack("<i",3))
    headdata.write(struct.pack("<i",len(meshes)))
    headdata.write(struct.pack("<i",0))
    headdata.write(struct.pack("<i",0))
    if armature:
        headdata.write(struct.pack("<i",len(armature.bones.values())))
    else:
        headdata.write(struct.pack("<i",0))
    
    headdata.write(struct.pack("<6f",*generate_bbox([x for _,x in meshes]))) # bounding box

    meshindex = 0

    file = open(filename, 'wb')
    file.write(b'SMDL')
    
    write_block(file,'head',headdata.getvalue())
    write_block(file,'bone',bonedata)


    for obj,mesh in meshes:
        vert = [vert for vert in mesh.verts]
        faces = []
        ftexc = []
        ftang = []
        fcolr = []
        flnk1 = []
        #faces = [[v.index for v in f.verts] for f in mesh.faces]
        #ftang = [[
        #ftexc = mesh.uv_layers.active.data
        
        #if mesh.vertexColors:
            #fcolr = [[c for c in f.col] for f in mesh.faces]
        #else:
            #fcolr = None
        uv_lay = mesh.loops.layers.uv.active
        if not uv_lay:
            ftexc = None
        col_lay = mesh.loops.layers.color.active
        if not col_lay:
            fcolr = None
        dvert_lay = mesh.verts.layers.deform.active

        flnk1 = [vert[dvert_lay].items() for vert in mesh.verts]

        for f in mesh.faces:
            uv = []
            col = []
            vindex = []
            tang = []
            for loop in f.loops:
                if ftexc is not None:
                    uv.append(loop[uv_lay].uv)
                vindex.append(loop.vert.index)
                if fcolr is not None:
                    col.append(loop[col_lay].color)
                tang.append(loop.calc_tangent())
            if ftexc is not None:
                ftexc.append(uv)
            ftang.append(tang)
            faces.append(vindex)
            if fcolr:
                fcolr.append(col)

        #duplication
        if ftexc:
            #texc = face_to_vertices_dup(faces,ftexc,vert)
            fsign = calcFaceSigns(ftexc)
            #duplication
            #sign = face_to_vertices_dup(faces,fsign,vert)
            sign = face_to_vertices(faces,fsign,vert)
            #recreate texc data due to duplicated vertices
            texc = face_to_vertices(faces,ftexc,vert)
            tang = face_to_vertices(faces,ftang,vert)
        #Gram-Schmidt orthogonalize
        for i in range(len(vert)):
            #tang[i] = (tang[i] - vert[i].normal * DotVecs(tang[i],vert[i].normal)).normalize()
            tang[i] = (tang[i] - vert[i].normal * tang[i].dot(vert[i].normal))
            tang[i].normalize()

        #lnk1 = face_to_vertices(faces,flnk1,vert)
        lnk1 = flnk1
        if fcolr is not None:
            colr = face_to_vertices(faces,fcolr,vert)
        else:
            colr = None
            
        write_block(file,'mesh',create_mesh_data(mesh,vert,meshindex,obj.name.encode('utf8'),obj.data.materials[0].name.encode('utf8')))
        write_block(file,'vrts',create_vrts_data(vert,meshindex))
        new_indices = {}
        print(bone_indices)
        for group in obj.vertex_groups:
            new_indices[group.index] = bone_indices.index(group.name)
        write_block(file,'lnk1',create_lnk1_data(lnk1,meshindex,new_indices))
        if len(faces) > 0:
            write_block(file,'face',create_face_data(vert,faces,meshindex))
            if ftexc is not None:
                write_block(file,"texc",create_texc_data(texc,meshindex))
                for i in range(len(tang)):
                    if sign[i] == 0:
                        tang[i] = -(tang[i].copy())
                write_block(file,"tang",create_tang_data(tang,meshindex))
                write_block(file,"sign",create_sign_data(meshindex,sign))
            write_block(file,"nrml",create_nrml_data(vert,meshindex))
        if fcolr is not None:
            write_block(file,"colr",create_colr_data(colr,meshindex))
        meshindex+=1
        vlog('total vertices duplicated: %d' % (len(vert) - len(mesh.verts)))
############################## 
#CLIPS
##############################
    
MKEY_X,MKEY_Y,MKEY_Z,\
MKEY_PITCH,MKEY_ROLL,MKEY_YAW,\
MKEY_VISIBILITY,\
MKEY_SCALE_X,MKEY_SCALE_Y,MKEY_SCALE_Z, \
MKEY_COUNT \
    = range(11)

from math import sqrt,atan2,degrees

def ClipBone(file,bone_name,motion,index):
    for keytype in range(MKEY_COUNT):
        keydata = BytesIO()
        key = motion[keytype]
        #if keytype != MKEY_VISIBILITY:
            #key = map(lambda k: round(k,ROUND_KEYS), key)
        if min(key) == max(key):
            key = [key[0]]
        numkeys = len(key)
        keydata.write(struct.pack("<i",index))
        keydata.write(struct.pack("<i",keytype))
        keydata.write(struct.pack("<i",numkeys))
        keydata.write(struct.pack("B",len(bone_name)))
        keydata.write(bone_name)
        keydata.write(struct.pack("B",0))
        if keytype == MKEY_VISIBILITY:
            keydata.write(struct.pack('%dB' % numkeys,*key))
        else:
            keydata.write(struct.pack('<%df' % numkeys,*key))
        write_block(file,'bmtn',keydata.getvalue())

def export_k2_clip(filename, transform,frame_start,frame_end):
    objList = bpy.context.selected_objects
    if len(objList) != 1 or objList[0].type != 'ARMATURE':
        err('Select needed armature only')
        return
    armob = objList[0]
    motions = {}
    vlog ('baking animation')
    armature = armob.data
    if transform:
        worldmat = armob.matrix_world
    else:
        worldmat = Matrix([1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1])
    scene = bpy.context.scene
    pose = armob.pose


    for frame in range(frame_start,frame_end):
        scene.frame_set(frame)
        for bone in pose.bones.values():
            matrix = bone.matrix
            if bone.parent:
                matrix = matrix * (bone.parent.matrix.copy().inverted())
            else:
                matrix = matrix * worldmat
            if bone.name not in motions:
                motions[bone.name] = []
                for i in range(MKEY_COUNT):
                    motions[bone.name].append([])
            motion = motions[bone.name]

            translation = matrix.translation
            rotation = matrix.to_euler('YXZ')
            scale = matrix.to_scale()
            visibility = 255
            
            motion[MKEY_X].append(translation[0])
            motion[MKEY_Y].append(translation[1])
            motion[MKEY_Z].append(translation[2])
            
            motion[MKEY_PITCH].append(-degrees(rotation[0]))
            motion[MKEY_ROLL].append(-degrees(rotation[1]))
            motion[MKEY_YAW].append(-degrees(rotation[2]))
            
            motion[MKEY_SCALE_X].append(scale[0])
            motion[MKEY_SCALE_Y].append(scale[1])
            motion[MKEY_SCALE_Z].append(scale[2])
            
            motion[MKEY_VISIBILITY].append(visibility)
            
    headdata = BytesIO()
    headdata.write(struct.pack("<i",2))
    headdata.write(struct.pack("<i",len(motions.keys())))
    headdata.write(struct.pack("<i",frame_end - frame_start))

    file = open(filename, 'wb')
    file.write(b'CLIP')
    write_block(file,'head',headdata.getvalue())
    
    index = 0
    for bone_name in sorted(armature.bones.keys(),key=lambda x:bone_depth(armature.bones[x])):
        ClipBone(file,bone_name.encode('utf8'),motions[bone_name],index)
        index+=1
    file.close()

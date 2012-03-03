#!BPY

""" Registration info for Blender menus:
Name: 'K2/HON (.model/.clip)...'
Blender: 249
Group: 'Export'
Tip: 'Export a Heroes of Newerth model file'
"""

__author__ = "Anton Romanov"
__version__ = "2010.03.24"

__bpydoc__ = """\
"""

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



import Blender,bpy
import struct, chunk
import cStringIO
from Blender.Mathutils import *
from Blender import Mesh, Scene, Window, sys, Image, Draw
import BPyMesh
import BPyObject
# determines the verbosity of loggin.
#   0 - no logging (fatal errors are still printed)
#   1 - standard logging
#   2 - verbose logging
#   3 - debug level. really boring (stuff like vertex data and verbatim lines)
IMPORT_LOG_LEVEL = 3

def log(msg):
    if IMPORT_LOG_LEVEL >= 1: print msg

def vlog(msg):
    if IMPORT_LOG_LEVEL >= 2: print msg

def dlog(msg):
    if IMPORT_LOG_LEVEL >= 3: print msg
    
def err(msg):
    Blender.Draw.PupMenu('Error|%s' % msg)
    
def bone_depth(bone):
    if not bone.parent:
        return 0
    else:
        return 1+bone_depth(bone.parent)
def create_bone_data(armature,armob):
    bones = []
    for bone in sorted(armature.bones.values(),cmp=lambda x, y: bone_depth(x) - bone_depth(y)):
        bones.append(bone.name)
    bonedata = cStringIO.StringIO()
    
    for name in bones:
        bone = armature.bones[name]
        base = bone.matrix['ARMATURESPACE'].copy()
        if transform:
            base *= armob.mat
        baseInv = base.copy().invert()
        if bone.parent :
            parent_index = bones.index(bone.parent.name)
        else:
            parent_index = -1
        #parent bone index
        bonedata.write(struct.pack("<i",parent_index))
        #inverted matrix
        bonedata.write(struct.pack('<12f', *sum([row[0:3] for row in baseInv],[])))
        #base matrix
        bonedata.write(struct.pack('<12f', *sum([row[0:3] for row in base],[])))
        #bone name
        bonedata.write(struct.pack("B" ,len(name)))
        bonedata.write(name)
        bonedata.write(struct.pack("B" ,0))
    return bones,bonedata.getvalue()
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
    
def create_mesh_data(mesh,vert,index):
    meshdata = cStringIO.StringIO()
    meshdata.write(struct.pack("<i",index))
    meshdata.write(struct.pack("<i",1)) # mode? huh? dunno...
    meshdata.write(struct.pack("<i",len(vert))) # vertices count
    meshdata.write(struct.pack("<6f",*generate_bbox([mesh]))) # bounding box
    meshdata.write(struct.pack("<i",-1)) # bone link... dunno... TODO
    meshdata.write(struct.pack("<B",len(mesh.name))) 
    meshdata.write(struct.pack("<B",len(mesh.materials[0].name))) 
    meshdata.write(mesh.name)
    meshdata.write(struct.pack("<B",0)) 
    meshdata.write(mesh.materials[0].name)
    meshdata.write(struct.pack("<B",0)) 
    return meshdata.getvalue()
    
def create_vrts_data(verts,meshindex):
    data = cStringIO.StringIO()
    data.write(struct.pack("<i",meshindex))
    for v in verts:
        data.write(struct.pack("<3f",*v.co))
    return data.getvalue()
    
def create_face_data(verts,faces,meshindex):
    data = cStringIO.StringIO()
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
    data = cStringIO.StringIO()
    data.write(struct.pack("<i",meshindex))
    data.write(struct.pack("<i",0)) # huh?
    for t in tang:
        data.write(struct.pack('<3f',*list(t)))
    return data.getvalue()

def write_block(file,name,data):
    file.write(name)
    file.write(struct.pack("<i",len(data)))
    file.write(data)

def create_texc_data(texc,meshindex):
    if flip_uv:
        for i in xrange(len(texc)):
            texc[i] = [texc[i][0],1.0-texc[i][1]]
    data = cStringIO.StringIO()
    data.write(struct.pack("<i",meshindex))
    data.write(struct.pack("<i",0)) # huh?
    for t in texc:
        data.write(struct.pack("<2f",*t))
    return data.getvalue()

def create_colr_data(colr,meshindex):
    data = cStringIO.StringIO()
    data.write(struct.pack("<i",meshindex))
    for c in colr:
        data.write(struct.pack("<4B",c.r,c.g,c.b,c.a))
    return data.getvalue()

def create_nrml_data(verts,meshindex):
    data = cStringIO.StringIO()
    data.write(struct.pack("<i",meshindex))
    for v in verts:
        data.write(struct.pack("<3f",*v.no))
    return data.getvalue()

def create_lnk1_data(lnk1,meshindex,bone_indices):
    data = cStringIO.StringIO()
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
                *[bone_indices.index(inf[0]) for inf in influences]))
    return data.getvalue()

def create_sign_data(meshindex,sign):
    data = cStringIO.StringIO()
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
def duplicate_twosided_faces(mesh,faces,vert,ftexc,flnk1,ftang,fcolr):
    for fi in [f.index for f in mesh.faces if f.mode & Blender.NMesh.FaceModes['TWOSIDE']]:
        newf = []
        for v in faces[fi]:
            newv = len(vert)
            newf.append(newv)
            vert.append(Blender.Mesh.MVert(vert[v].co))
            vert[newv].no = -(vert[v].no.copy())
        newf.reverse()
        faces.append(newf)
        for ar in [ar for ar in [ftexc,flnk1,ftang,fcolr] if ar is not None]:
            ar.append(ar[fi])
            if type(ar[-1]) == type(list()):
                ar[-1].reverse()
            else:
                tmp = list(ar[-1])
                tmp.reverse()
                ar[-1] = tuple(tmp)

def CreateK2Mesh(filename):
    
    if not filename.lower().endswith('.model'):
        filename += '.model'
    scn = Blender.Scene.GetCurrent()
    
    #meshes = [ob for ob in scn.objects if ob.type == 'Mesh']
    meshes = []
    armOb = [ob for ob in scn.objects if ob.type == 'Armature'][0]
    armature = armOb.getData()
    rest = armature.restPosition
    armature.restPosition = True
    armOb.makeDisplayList()
    # This causes the makeDisplayList command to effect the mesh
    Blender.Set('curframe', Blender.Get('curframe'))

    vlog ("==========================export start=======================================")
    for ob_main in [ob for ob in scn.objects if ob.type == 'Mesh']:
        for ob, ob_mat in BPyObject.getDerivedObjects(ob_main):
            mesh = BPyMesh.getMeshFromObject(ob, None, EXPORT_APPLY_MODIFIERS, True, scn)
            if not mesh:
                continue
            #triangulate
            mesh.name = ob.name
            has_quads = False
            for f in mesh.faces:
                if len(f) == 4:
                    has_quads = True
                    break
            if has_quads:
                vlog('triangulating')
                oldmode = Mesh.Mode()
                Mesh.Mode(Mesh.SelectModes['FACE'])
                mesh.sel = True
                tempob = scn.objects.new(mesh)
                mesh.quadToTriangle(0) # more=0 shortest length
                oldmode = Mesh.Mode(oldmode)
                scn.objects.unlink(tempob)
                Mesh.Mode(oldmode)
            #copy normals
            for v in ob.getData(mesh=1).verts:
                mesh.verts[v.index].no = v.no
            
            mesh.transform(ob_mat)
            
            # High Quality Normals
            if hq_normals:
                BPyMesh.meshCalcNormals(mesh)
            else:
                # transforming normals is incorrect
                # when the matrix is scaled,
                # better to recalculate them
                mesh.calcNormals()
            meshes.append(mesh)
    
    bone_indices,bonedata = create_bone_data(armature,armOb)
    
    headdata = cStringIO.StringIO()
    headdata.write(struct.pack("<i",3))
    headdata.write(struct.pack("<i",len(meshes)))
    headdata.write(struct.pack("<i",0))
    headdata.write(struct.pack("<i",0))
    headdata.write(struct.pack("<i",len(armature.bones.values())))
    
    headdata.write(struct.pack("<6f",*generate_bbox(meshes))) # bounding box

    meshindex = 0

    file = open(filename, 'wb')
    file.write('SMDL')
    
    write_block(file,'head',headdata.getvalue())
    write_block(file,'bone',bonedata)

    for mesh in meshes:
        vert = [vert for vert in mesh.verts]
        flnk1 = [[mesh.getVertexInfluences(v.index) for v in f.v] for f in mesh.faces]
        faces = [[v.index for v in f.v] for f in mesh.faces]
        ftang = mesh.getTangents()
        ftexc = [[(round(uv.x, 6), round(uv.y, 6)) for uv in f.uv] for f in mesh.faces]
        
        if mesh.vertexColors:
            fcolr = [[c for c in f.col] for f in mesh.faces]
        else:
            fcolr = None

        duplicate_twosided_faces(mesh,faces,vert,ftexc,flnk1,ftang,fcolr)

        if mesh.faceUV:
            #duplication
            texc = face_to_vertices_dup(faces,ftexc,vert)
            if hq_lighting:
                fsign = calcFaceSigns(ftexc)
                #duplication
                sign = face_to_vertices_dup(faces,fsign,vert)
                #recreate texc data due to duplicated vertices
                texc = face_to_vertices(faces,ftexc,vert)
                
        tang = face_to_vertices(faces,ftang,vert)
        #Gram-Schmidt orthogonalize
        for i in xrange(len(vert)):
            tang[i] = (tang[i] - vert[i].no * DotVecs(tang[i],vert[i].no)).normalize()

        lnk1 = face_to_vertices(faces,flnk1,vert)
        if mesh.vertexColors:
            colr = face_to_vertices(faces,fcolr,vert)
        else:
            colr = None
            
        write_block(file,'mesh',create_mesh_data(mesh,vert,meshindex))
        write_block(file,'vrts',create_vrts_data(vert,meshindex))
        write_block(file,'lnk1',create_lnk1_data(lnk1,meshindex,bone_indices))
        if len(faces) > 0:
            write_block(file,'face',create_face_data(vert,faces,meshindex))
            if mesh.faceUV:
                write_block(file,"texc",create_texc_data(texc,meshindex))
                if hq_lighting:
                    for i in xrange(len(tang)):
                        if sign[i] == 0:
                            tang[i] = -(tang[i].copy())
                write_block(file,"tang",create_tang_data(tang,meshindex))
                if hq_lighting:
                    write_block(file,"sign",create_sign_data(meshindex,sign))
            write_block(file,"nrml",create_nrml_data(vert,meshindex))
        if mesh.vertexColors:
            write_block(file,"colr",create_colr_data(colr,meshindex))
        meshindex+=1
        vlog('total vertices duplicated: %d' % (len(vert) - len(mesh.verts)))
        
    armature.restPosition = rest
    armOb.makeDisplayList()
    # This causes the makeDisplayList command to effect the mesh
    Blender.Set('curframe', Blender.Get('curframe'))
def roundVector(vec,dec=17):
    fvec=[]
    for v in vec:
        fvec.append(round(v,dec))
    return Vector(fvec)
def roundMatrix(mat,dec=17):
    fmat = []
    for row in mat:
        fmat.append(roundVector(row,dec))
    return Matrix(*fmat)
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
from sys import float_info

def MatToEulerYXZ_(M):
    e = Vector(0.0,0.0,0.0)
    i = 1
    j = 0
    k = 2
    cy = sqrt(M[i][i]*M[i][i] + M[j][i]*M[j][i])
    if (cy > 16*float_info.epsilon):
        e[0] = atan2(M[j][k], M[k][k])
        e[1] = atan2(-M[i][k], cy)
        e[2] = atan2(M[i][j], M[i][i])
    else:
        e[0] = atan2(-M[k][j], M[j][j])
        e[1] = atan2(-M[i][k], cy)
        e[2] = 0
    e[0] = -e[0] 
    e[1] = -e[1] 
    e[2] = -e[2]
    return e
    
_epsilon = 1E-12

def _eulerIndices(i, neg, alt):
    """Helper function for _getRotation()."""
    next = [1, 2, 0, 1]
    j = next[i+int(neg)]
    k = 3-i-j
    h = next[k+(1^int(neg)^int(alt))]
    return j,k,h
def _eulerGivens(a, b):
    """Helper function for _getRotation()."""
    global _epsilon
    
    absa = abs(a)
    absb = abs(b)
    # b=0?
    if absb<=_epsilon:
        if a>=0:
            c = 1.0
        else:
            c = -1.0
        return (c, 0.0, absa)
    # a=0?
    elif absa<=_epsilon:
        if b>=0:
            s = 1.0
        else:
            s = -1.0
        return (0.0, s, absb)
    # General case
    else:
        if absb>absa:
            t = a/b
            u = sqrt(1.0+t*t)
            if b<0:
                u = -u
            s = 1.0/u
            c = s*t
            r = b*u
        else:
            t = b/a
            u = sqrt(1.0+t*t)
            if (a<0):
                u = -u
            c = 1.0/u
            s = c*t
            r = a*u
        return c,s,r

def MatToEulerYXZ(M):
    y,x,z = _getRotation(2, True, True, True,M)
    return Vector(x,y,z)
    
def _getRotation(i, neg, alt, rev,M):
    i = 2
    neg = True
    alt = True
    rev = True
    v = [M[0][i], M[1][i], M[2][i]]
    j,k,h = _eulerIndices(i, neg, alt)
    a = v[h]
    b = v[k]
    c,s,r = _eulerGivens(a, b)
    v[h] = r
    s1 = c*M[k][j] - s*M[h][j]
    c1 = c*M[k][k] - s*M[h][k]
    r1 = atan2(s1, c1)
    r2 = atan2(v[j], v[i])
    r3 = atan2(s, c)
    if alt:
        r3 = -r3
    if neg:
        r1 = -r1
        r2 = -r2
        r3 = -r3
    if rev:
        tmp = r1
        r1 = r3
        r3 = tmp
    return r1,r2,r3
    
def ClipBone(file,bone_name,motion,index):
    for keytype in xrange(MKEY_COUNT):
        keydata = cStringIO.StringIO()
        key = motion[keytype]
        if keytype != MKEY_VISIBILITY:
            key = map(lambda k: round(k,ROUND_KEYS), key)
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
        
def CreateK2Clip(filename):
    objList = Blender.Object.GetSelected()
    if len(objList) != 1 or objList[0].type != 'Armature':
        err('Select needed armature only')
        return
    armob = objList[0]
    motions = {}
    vlog ('baking animation')
    armature = armob.getData()
    if transform:
        worldmat = armob.mat.copy()
    else:
        worldmat = Matrix([1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1])

    for frame in range(startframe,endframe):
        armob.evaluatePose(frame)
        pose = armob.getPose()
        for bone in pose.bones.values():
            matrix = bone.poseMatrix
            if bone.parent:
                matrix = matrix * (bone.parent.poseMatrix.copy().invert())
            else:
                matrix = matrix * worldmat
            if bone.name not in motions:
                motions[bone.name] = []
                for i in xrange(MKEY_COUNT):
                    motions[bone.name].append([])
            motion = motions[bone.name]

            translation = matrix.translationPart()
            rotation = MatToEulerYXZ(matrix.rotationPart())
            scale = matrix.scalePart()
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
            
    headdata = cStringIO.StringIO()
    headdata.write(struct.pack("<i",2))
    headdata.write(struct.pack("<i",len(motions.keys())))
    headdata.write(struct.pack("<i",endframe - startframe))

    file = open(filename, 'wb')
    file.write('CLIP')
    write_block(file,'head',headdata.getvalue())
    
    index = 0
    for bone_name in sorted(armature.bones.keys(),cmp=lambda x, y: bone_depth(armature.bones[x]) - bone_depth(armature.bones[y])):
        ClipBone(file,bone_name,motions[bone_name],index)
        index+=1
    #file.close()
    
#-------------------------------\
#           common           |
#-------------------------------/
######################################################
# GUI STUFF
######################################################

draw_busy_screen = 0
EVENT_NOEVENT = 1
EVENT_EXPORT = 2
EVENT_QUIT = 3
EVENT_MESHFILENAME = 4
EVENT_ANIMFILENAME = 5
EVENT_MESHFILENAME_STRINGBUTTON = 6
EVENT_ANIMFILENAME_STRINGBUTTON = 7
EVENT_APPLY_TRANSFORM = 7
mesh_filename = Blender.Draw.Create("")
anim_filename = Blender.Draw.Create("")
apply_transform = True
EXPORT_APPLY_MODIFIERS = Blender.Draw.Create(1)
EXPORT_HQ_NORMALS = Blender.Draw.Create(1)
EXPORT_HQ_LIGHTING = Blender.Draw.Create(1)
EXPORT_FLIP_UV = Blender.Draw.Create(1)


startframe = 1
endframe = Blender.Scene.GetCurrent().getRenderingContext().endFrame() + 1
STATRFRAME_slider = Blender.Draw.Create(startframe)
ENDFRAME_slider = Blender.Draw.Create(endframe - 1)



flip_uv = True
scale = 1.0
hq_normals = True
transform = True
hq_lighting = True
ROUND_KEYS = 4

ROUNDING_slider = Blender.Draw.Create(ROUND_KEYS)
######################################################
# Callbacks for Window functions
######################################################
def meshname_callback(filename):
  global mesh_filename
  mesh_filename.val=filename

def animname_callback(filename):
  global anim_filename
  #anim_filename.val=Blender.sys.dirname(filename)
  anim_filename.val=filename

######################################################
# GUI Functions
######################################################
def handle_event(evt, val):
  if evt == Blender.Draw.ESCKEY:
    Blender.Draw.Exit()
    return

def handle_button_event(evt):
  global EVENT_NOEVENT, EVENT_EXPORT, EVENT_QUIT, EVENT_MESHFILENAME, EVENT_ANIMFILENAME, EVENT_MESHFILENAME_STRINGBUTTON, EVENT_ANIMFILENAME_STRINGBUTTON,EXPORT_FLIP_UV
  global flip_uv,draw_busy_screen, mesh_filename, anim_filename, scale_slider, scale,EXPORT_HQ_NORMALS,hq_normals,EXPORT_APPLY_MODIFIERS,transform,EXPORT_HQ_LIGHTING,hq_lighting
  global STATRFRAME_slider,startframe,ENDFRAME_slider,endframe,ROUND_KEYS,ROUNDING_slider
  if evt == EVENT_EXPORT:
    transform = EXPORT_APPLY_MODIFIERS.val
    hq_normals = EXPORT_HQ_NORMALS.val
    hq_lighting = EXPORT_HQ_LIGHTING.val
    flip_uv = EXPORT_FLIP_UV.val
    Blender.Window.WaitCursor(1)
    draw_busy_screen = 1
    startframe = STATRFRAME_slider.val
    endframe = ENDFRAME_slider.val + 1
    ROUND_KEYS = ROUNDING_slider.val
    Blender.Draw.Draw()
    if len(mesh_filename.val)>0:
        CreateK2Mesh(mesh_filename.val)
    if len(anim_filename.val)>0:
        CreateK2Clip(anim_filename.val)
    draw_busy_screen = 0
    Blender.Draw.Redraw(1)
    Blender.Window.WaitCursor(0)
    return
  if evt == EVENT_QUIT:
    Blender.Draw.Exit()
  if evt == EVENT_MESHFILENAME:
    Blender.Window.FileSelector(meshname_callback, "Select mesh file...")
    Blender.Draw.Redraw(1)
  if evt == EVENT_ANIMFILENAME:
    Blender.Window.FileSelector(animname_callback, "Select anim file...")
    Blender.Draw.Redraw(1)

def show_gui():
  global EVENT_NOEVENT, EVENT_EXPORT, EVENT_QUIT, EVENT_MESHFILENAME, EVENT_ANIMFILENAME, EVENT_MESHFILENAME_STRINGBUTTON, EVENT_ANIMFILENAME_STRINGBUTTON,EXPORT_FLIP_UV
  global draw_busy_screen, mesh_filename, anim_filename, scale_slider,EXPORT_APPLY_MODIFIERS,EXPORT_HQ_NORMALS,EXPORT_HQ_LIGHTING
  global STATRFRAME_slider,startframe,ENDFRAME_slider,endframe,ROUND_KEYS,ROUNDING_slider
  button_width = 240
  browsebutton_width = 60
  button_height = 25
  if draw_busy_screen == 1:
    Blender.BGL.glClearColor(0.3,0.3,0.3,1.0)
    Blender.BGL.glClear(Blender.BGL.GL_COLOR_BUFFER_BIT)
    Blender.BGL.glColor3f(1,1,1)
    Blender.BGL.glRasterPos2i(20,25)
    Blender.Draw.Text("Please wait...")
    return
  Blender.BGL.glClearColor(0.6,0.6,0.6,1.0)
  Blender.BGL.glClear(Blender.BGL.GL_COLOR_BUFFER_BIT)
  Blender.Draw.Button("Export!", EVENT_EXPORT, 20, 2*button_height, button_width, button_height, "Start the export")
  Blender.Draw.Button("Quit", EVENT_QUIT, 20, button_height, button_width, button_height, "Quit this script")
  
  Blender.Draw.Button("Browse...", EVENT_MESHFILENAME, 21+button_width-browsebutton_width, 8*button_height, browsebutton_width, button_height, "Specify mesh-file")
  Blender.Draw.Button("Browse...", EVENT_ANIMFILENAME, 21+button_width-browsebutton_width, 3*button_height, browsebutton_width, button_height, "Specify anim-file")
  mesh_filename = Blender.Draw.String("Mesh file:", EVENT_MESHFILENAME_STRINGBUTTON, 20, 8*button_height, button_width-browsebutton_width, button_height, mesh_filename.val, 255, "Mesh-File to export")
  anim_filename = Blender.Draw.String("Anim file:", EVENT_ANIMFILENAME_STRINGBUTTON, 20, 3*button_height, button_width-browsebutton_width, button_height, anim_filename.val, 255, "Anim-File to export")
  EXPORT_FLIP_UV = Blender.Draw.Toggle('V-Flip UV', EVENT_NOEVENT, 20, 9*button_height, button_width, button_height, EXPORT_FLIP_UV.val, 'Flip UV mapping vertically.')
  EXPORT_APPLY_MODIFIERS = Blender.Draw.Toggle('Apply Modifiers', EVENT_NOEVENT, 20, 10*button_height, button_width, button_height, EXPORT_APPLY_MODIFIERS.val, 'Use transformed mesh data from each object.')
  EXPORT_HQ_NORMALS = Blender.Draw.Toggle('HQ Normals', EVENT_NOEVENT ,20, 11*button_height, button_width, button_height,  EXPORT_HQ_NORMALS.val, 'Calculate high quality normals for rendering.')
  EXPORT_HQ_LIGHTING = Blender.Draw.Toggle('HQ Lighting', EVENT_NOEVENT ,20, 12*button_height, button_width, button_height,  EXPORT_HQ_LIGHTING.val, 'Proper saving when mesh uses mirrored normal map. Increases vertex count.. etc')

  ROUNDING_slider = Blender.Draw.Number("Round Motion Keys to:", EVENT_NOEVENT, 20, 6*button_height, button_width, button_height, ROUNDING_slider.val, 0,20, "How many digits leave after the decimal point in float motion keys")
  STATRFRAME_slider = Blender.Draw.Number("Start Frame:", EVENT_NOEVENT, 20, 5*button_height, button_width, button_height, STATRFRAME_slider.val, 1,1000, "Set starting frame")
  ENDFRAME_slider = Blender.Draw.Number("End Frame:", EVENT_NOEVENT, 20, 4*button_height, button_width, button_height, ENDFRAME_slider.val, 1, 1000, "Set ending frame")
Blender.Draw.Register (show_gui, handle_event, handle_button_event)

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

__author__ = ["Anton Romanov"]
__version__ = '1.0'
__bpydoc__ = """"""

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

def parse_links(honchunk,mesh,bone_names):
    mesh_index = read_int(honchunk)
    numverts = read_int(honchunk)
    dlog("links")
    dlog("mesh index: %d" % mesh_index)
    dlog("vertices number: %d" % numverts)
    for i in xrange(numverts):
        num_weights = read_int(honchunk)
        weights = struct.unpack("<%df" % num_weights,honchunk.read(num_weights * 4))
        indexes = struct.unpack("<%dI" % num_weights,honchunk.read(num_weights * 4))
        for ii, index in enumerate(indexes):
            name = bone_names[index]
            if name not in mesh.getVertGroupNames():
                mesh.addVertGroup(name)
            mesh.assignVertsToGroup(name,[i],weights[ii],Blender.Mesh.AssignModes.ADD)
    honchunk.skip()

def parse_vertices(honchunk):
    vlog('parsing vertices chunk')
    numverts = (honchunk.chunksize - 4)/12
    vlog('%d vertices' % numverts)
    meshindex = read_int(honchunk)
    return [struct.unpack("<3f", honchunk.read(12)) for i in xrange(numverts)]
def parse_sign(honchunk):
    vlog('parsing sign chunk')
    numverts = (honchunk.chunksize - 8)
    meshindex = read_int(honchunk)
    vlog(read_int(honchunk)) # huh?
    return [struct.unpack("<b", honchunk.read(1)) for i in xrange(numverts)]
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
        return [struct.unpack("<3H", honchunk.read(6)) for i in xrange(numfaces)]
    elif size ==1:
        return [struct.unpack("<3B", honchunk.read(3)) for i in xrange(numfaces)]
    elif size == 4:
        return [struct.unpack("<3I", honchunk.read(12)) for i in xrange(numfaces)]
    else:
        log("unknown size for faces:%d" % size)
        return []
def parse_normals(honchunk):
    vlog('parsing normals chunk')
    numverts = (honchunk.chunksize - 4)/12
    vlog('%d normals' % numverts)
    meshindex = read_int(honchunk)
    return [struct.unpack("<3f", honchunk.read(12)) for i in xrange(numverts)]
def parse_texc(honchunk,version):
    vlog('parsing uv texc chunk')
    numverts = (honchunk.chunksize - 4)/8
    vlog('%d texc' % numverts)
    meshindex = read_int(honchunk)
    if version == 3:
        vlog(read_int(honchunk)) # huh?
    return [struct.unpack("<2f", honchunk.read(8)) for i in xrange(numverts)]   
def parse_colr(honchunk):
    vlog('parsing vertex colours chunk')
    numverts = (honchunk.chunksize - 4)/4
    meshindex = read_int(honchunk)
    return [struct.unpack("<4B", honchunk.read(4)) for i in xrange(numverts)]
    

def CreateBlenderMesh(filename,objName):
    file = open(filename,'rb')
    if not file:
        log("can't open file")
        return
    sig = file.read(4)
    if sig != b'SMDL':
        err('unknown file signature')
        err(sig)
        return

    try:
        honchunk = chunk.Chunk(file,bigendian=0,align=0)
    except EOFError:
        log('error reading first chunk')
        return
    if honchunk.chunkname != b'head':
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

    scn= bpy.data.scenes.active

    try:
        honchunk = chunk.Chunk(file,bigendian=0,align=0)
    except EOFError:
        log('error reading bone chunk')
        return
        
    
    #read bones

    #create armature object


def CreateBlenderClip(filename,objName):
    dlog('Importing clip')

class K2Importer(bpy.types.Operator):
    '''Load K2/Silverlight mesh/clip data'''
    bl_idname = "import_mesh.k2"
    bl_label = "Import k2"

    filepath = StringProperty(name="File Path", description="Filepath used for importing the k2 file", maxlen=1024, default="", subtype='FILE_PATH')

    def execute(self, context):
        dlog(self.filepath)

        #convert the filename to an object name
        objName = bpy.path.display_name_from_filepath(self.filepath)
        dlog(objName)
        if self.filepath.endswith('.clip'):
            CreateBlenderClip(self.filepath,objName)
        else:
            CreateBlenderMesh(self.filepath,objName)

        #mesh = readMesh(self.filepath, objName)
        #addMeshObj(mesh, objName)

        return {'FINISHED'}

    def invoke(self, context, event):
        wm = context.window_manager
        wm.fileselect_add(self)
        return {'RUNNING_MODAL'}

# package manages registering

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

# <pep8-80 compliant>


bl_info = {
    "name": "K2 Model/Animation Import-Export",
    "author": "Anton Romanov",
    "version": (0, 1),
    "blender": (2, 6, 3),
    "location": "File > Import-Export > K2 model/clip",
    "description": "Import-Export meshes and animations used by K2 engine (Savage 2 and Heroes of Newerth games)",
    "warning": "",
    "wiki_url": "https://github.com/theli-ua/K2-Blender/wiki",
    "tracker_url": "https://github.com/theli-ua/K2-Blender/issues",
    "category": "Import-Export"}

if "bpy" in locals():
    import imp
    if "k2_import" in locals():
        imp.reload(k2_import)
    if "k2_export" in locals():
        imp.reload(k2_export)
else:
    import bpy

from bpy.props import StringProperty, BoolProperty

class K2ImporterClip(bpy.types.Operator):
    '''Load K2/Silverlight clip data'''
    bl_idname = "import_clip.k2"
    bl_label = "Import K2 Clip"

    filepath = StringProperty(
            subtype='FILE_PATH',
            )
    filter_glob = StringProperty(default="*.clip", options={'HIDDEN'})

    def execute(self, context):
        from . import k2_import
        k2_import.readclip(self.filepath)
        return {'FINISHED'}

    def invoke(self, context, event):
        wm = context.window_manager
        wm.fileselect_add(self)
        return {'RUNNING_MODAL'}

class K2Importer(bpy.types.Operator):
    '''Load K2/Silverlight mesh data'''
    bl_idname = "import_mesh.k2"
    bl_label = "Import K2 Mesh"

    filepath = StringProperty(
            subtype='FILE_PATH',
            )
    filter_glob = StringProperty(default="*.model", options={'HIDDEN'})

    def execute(self, context):
        from . import k2_import
        k2_import.read(self.filepath)
        return {'FINISHED'}

    def invoke(self, context, event):
        wm = context.window_manager
        wm.fileselect_add(self)
        return {'RUNNING_MODAL'}

class K2MeshExporter(bpy.types.Operator):
    '''Save K2 triangle mesh data'''
    bl_idname = "export_mesh.k2"
    bl_label = "Export K2 Mesh"

    filepath = StringProperty(
            subtype='FILE_PATH',
            )
    filter_glob = StringProperty(default="*.model", options={'HIDDEN'})
    check_existing = BoolProperty(
            name="Check Existing",
            description="Check and warn on overwriting existing files",
            default=True,
            options={'HIDDEN'},
            )
    apply_modifiers = BoolProperty(
            name="Apply Modifiers",
            description="Use transformed mesh data from each object",
            default=True,
            )

    def execute(self, context):
        from . import k2_export
        k2_export.export_k2_mesh(self.filepath,
                self.apply_modifiers
                         )

        return {'FINISHED'}

    def invoke(self, context, event):
        if not self.filepath:
            self.filepath = bpy.path.ensure_ext(bpy.data.filepath, ".model")
        wm = context.window_manager
        wm.fileselect_add(self)
        return {'RUNNING_MODAL'}


def menu_import(self, context):
    self.layout.operator(K2Importer.bl_idname, text="K2 mesh (.model)")
    self.layout.operator(K2ImporterClip.bl_idname, text="K2 clip (.clip)")


def menu_export(self, context):
    self.layout.operator(K2MeshExporter.bl_idname, text="K2 Mesh (.model)")


def register():
    bpy.utils.register_module(__name__)

    bpy.types.INFO_MT_file_import.append(menu_import)
    bpy.types.INFO_MT_file_export.append(menu_export)


def unregister():
    bpy.utils.unregister_module(__name__)

    bpy.types.INFO_MT_file_import.remove(menu_import)
    bpy.types.INFO_MT_file_export.remove(menu_export)

if __name__ == "__main__":
    register()

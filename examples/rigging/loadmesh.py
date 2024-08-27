import os.path
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pyglet
import pyglet.gl as GL
from pyglet.gl import *
import ctypes
import trimesh as tm

if sys.path[0] != "":
    sys.path.insert(0, "")

import struct
from pygltflib import GLTF2

from grafica.arcball import Arcball
from grafica.textures import texture_2D_setup

g_bones = []

accessorComponentType = {5120 : "b",
                         5121 : "B",
                         5122 : "h",
                         5123 : "H",
                         5125 : "I",
                         5126 : "f"}

accessorComponentTypeSize = {"b" : 1,
                         "B" : 1,
                         "h" : 2,
                         "H" : 2,
                         "I" : 4,
                         "f" : 4}

accesorType = {
    "SCALAR" : 1,
    "VEC2" : 2,
    "VEC3" : 3,
    "VEC4" : 4,
    "MAT2" : 4,
    "MAT3" : 9,
    "MAT4" : 16,
}

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname((os.path.abspath(__file__)))))
)

import grafica.transformations as tr
from grafica.utils import load_pipeline

def recursive_add_node(graph, json, count, axis, axis_pipeline, escale = 1, center = [0, 0, 0]):
    bone_list = json.skins[0].joints
    node = json.nodes[bone_list[count]]
    node_transform = tr.identity()
    length_in = 0.5
    length_out = 0.5
    if count == 0:
        node_transform = tr.translate(-center[0], -center[1], -center[2]) @ tr.uniformScale(escale)
    if node.scale:
        node_transform = tr.scale(*node.scale) @ node_transform 
    if node.rotation:
        qx = tm.transformations.quaternion_about_axis(node.rotation[0], [1, 0, 0])
        qy = tm.transformations.quaternion_about_axis(node.rotation[1], [0, 1, 0])
        qz = tm.transformations.quaternion_about_axis(node.rotation[2], [0, 0, 1])
        q = tm.transformations.quaternion_multiply(qx, qy)
        q = tm.transformations.quaternion_multiply(q, qz)
        node_transform = tm.transformations.quaternion_matrix(q) @ node_transform
    if node.translation:
        length_in = np.linalg.norm(tr.translate(*node.translation) - tr.translate(0,0,0))
        node_transform = tr.translate(*node.translation) @ node_transform
        #print (node.name, node.translation)
    
    graph.add_node(
        node.name,
        transform=node_transform,
    )
    #print (node.name, axis.position)
    
    if node.children:
        for child in node.children:
            graph, count, length_out = recursive_add_node(graph, json, count+1, axis, axis_pipeline)
            graph.add_edge(node.name, json.nodes[child].name)
            #print(node.name, "->", json.nodes[child].name)
    
    
    graph.add_node(
        "axis"+node.name,
        mesh=axis,
        pipeline=axis_pipeline,
        transform=tr.uniformScale(length_out),
        mode=GL.GL_LINES if node.mesh is None else GL.GL_TRIANGLES,
        color=np.array((1.0, 0.73, 0.03)) if not node.mesh is None else None
    )
    graph.add_edge(node.name, "axis"+node.name)
    print(node.name, length_out)
    return graph, count, length_in
    

def create_axis_graph(scene, axis, axis_pipeline):
    graph = nx.DiGraph(root="root")
    graph.add_node('root')

    return graph

def parentBones(gltf, joints, bones, vBones, count, parent):
    vBones.append(parent @ bones[count])
    for child in gltf.nodes[joints[count]].children:
        parentBones(gltf, joints, bones, vBones, count+1, parent @ bones[count])
    return vBones

def addJointDict(joints, cjoint, graph, axis, axis_pipeline, point_list, parent_point):
    
    if cjoint in joints:
        for node in joints[cjoint]:
            graph.add_node(
                node,
                transform= np.matrix(joints[cjoint][node]["matrix"]),
            )
            graph.add_node(
                "axis"+node,
                mesh=axis,
                pipeline=axis_pipeline,
                transform= tr.identity(),
                mode=GL.GL_LINES
            )
            graph.add_edge(cjoint, node)
            graph.add_edge(node, "axis"+node)
            point_list.append(parent_point @ np.array(joints[cjoint][node]["matrix"]))
            g_bones.append(np.array(joints[cjoint][node]["matrix"]))
            addJointDict(joints, node, graph, axis, axis_pipeline, point_list, parent_point @ np.matrix(joints[cjoint][node]["matrix"]))
    return graph

def addMeshToGraph(
    mesh, mesh_pipeline, graph, name
):   
    graph.add_node(
        name,
        mesh=mesh,
        pipeline=mesh_pipeline,
        transform=tr.identity(),
        mode=GL.GL_TRIANGLES,
        color=np.array((1.0, 0.73, 0.03)) 
    )

    graph.add_edge('root', name)

    return graph


def checkData(gltf, accessorNum):
    accessor_list = []
    accessor_type = accessorComponentType[gltf.accessors[accessorNum].componentType] * accesorType[gltf.accessors[accessorNum].type]
    accessor_bytes = accessorComponentTypeSize[accessorComponentType[gltf.accessors[accessorNum].componentType]] * accesorType[gltf.accessors[accessorNum].type]   
    bufferView = gltf.accessors[accessorNum].bufferView
    print(bufferView)
    data = gltf.get_data_from_buffer_uri(gltf.buffers[0].uri)
    for i in range(gltf.accessors[accessorNum].count):
        index = (
            gltf.bufferViews[bufferView].byteOffset
            + gltf.accessors[accessorNum].byteOffset
            + i * accessor_bytes
        )  # the location in the buffer of this vertex
        d = data[index : index + accessor_bytes]  # the vertex data
        v = struct.unpack("<" + accessor_type, d)  # convert from base64 to three floats
        if gltf.accessors[accessorNum].type[0] == "M":
            num = int(gltf.accessors[accessorNum].type[3])
            accessor_list.append(np.array(v).reshape(num, num))
        elif gltf.accessors[accessorNum].type[0] == "S":
            accessor_list.append(v[0])
        else: 
            accessor_list.append(v)
    return accessor_list
if __name__ == "__main__":

    width = 1080
    height = 1080

    window = pyglet.window.Window(width, height)

    filename = "assets/xbot/scene.gltf"
    triMesh = tm.load(filename)
    meshGraph = triMesh.graph.to_networkx()
    
    gltf = GLTF2().load(filename)
    rootJointNode = gltf.nodes[gltf.skins[0].joints[0]].name


    asset = triMesh
    
    
    # for object_id, object_geometry in asset.geometry.items():
    #     print("geo before", tm.rendering.mesh_to_vertexlist(object_geometry)[4][1])
    # de acuerdo a la documentación de trimesh, esto centra la escena
    # # no es igual a trabajar con una malla directamente
    scale = 1.0 / asset.scale 
    center = asset.centroid
    # asset.rezero()
    # y esto la escala. se crea una copia, por eso la asignación
    asset = asset.scaled(scale)
    # for object_id, object_geometry in asset.geometry.items():
    #     print("geo after", tm.rendering.mesh_to_vertexlist(object_geometry)[4][1])

    axis_positions = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1])
    axis_colors = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1])
    axis_indices = np.array([0, 1, 2, 3, 4, 5])

    axis_pipeline = load_pipeline(Path(os.path.dirname(__file__)) / "line_vertex_program.glsl", Path(os.path.dirname(__file__)) / ".." / "hello_world" / "fragment_program.glsl")

    axis_gpu = axis_pipeline.vertex_list_indexed(6, GL.GL_LINES, axis_indices)
    axis_gpu.position[:] = axis_positions
    axis_gpu.color[:] = axis_colors

    bindReverse = checkData(gltf, gltf.skins[0].inverseBindMatrices)
    jointnums = gltf.skins[0].joints


    graph = nx.DiGraph(root = "")
    graph.add_node(
        rootJointNode,
        transform= tr.identity(),
    )
    graph.add_node(
        "axis"+rootJointNode,
        mesh=axis_gpu,
        pipeline=axis_pipeline,
        transform= tr.identity(),
        mode=GL.GL_LINES
    )
    graph.add_edge("root", rootJointNode)
    graph.add_edge(rootJointNode, "axis"+rootJointNode)
    g_bones.append(tr.identity())
    bones = [tr.identity()]
    graph = addJointDict(meshGraph, rootJointNode, graph, axis_gpu, axis_pipeline, bones, tr.identity())

    cloud = tm.PointCloud(bones)

    g_bones = g_bones + ([tr.identity()] * (70 - len(g_bones)))

    vBindReverse = []
    # Calculamos la posición de cada hueso segun su reverse bind al punto de origen
    vBindReverse = parentBones(gltf, jointnums, bindReverse, vBindReverse, 0, bindReverse[0])

    vBindReverse = vBindReverse + ([tr.identity()] * (70 - len(vBindReverse)))

    v_bones = []

    for i in range(len(g_bones)):
        v_bones.append(g_bones[i] @ vBindReverse[i])
        

    print("v_nones:", len(v_bones))


    # g_bones = g_bones + ([np.ctypeslib.as_ctypes(np.copy((np.asarray(tr.identity().reshape(16, 1, order="F"))/255).flatten()).astype(np.float32))] * (70 - len(g_bones)))
    # g_bones2 = np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=1) (*g_bones)
    node_scale = 1.0/np.linalg.norm(cloud.bounds[0] - cloud.bounds[1])

# como no todos los archivos que carguemos tendrán textura,
    # tendremos dos pipelines
    tex_pipeline = load_pipeline(
        Path(os.path.dirname(__file__)) / "mesh_vertex.glsl",
        Path(os.path.dirname(__file__)) / "fragment_program.glsl",
    )

    notex_pipeline = load_pipeline(
        Path(os.path.dirname(__file__)) / "mesh_vertex_notex.glsl",
        Path(os.path.dirname(__file__)) / "fragment_program_notex.glsl",
    )

    # aquí guardaremos las mallas del modelo que graficaremos
    vertex_lists = {}
    # con esto iteramos sobre las mallas
    count = 0
    for object_id, object_geometry in asset.geometry.items():
        mesh = {}
        print(object_id)
        joints = []
        weights = []
        cDWeigths = checkData(gltf, gltf.meshes[count].primitives[0].attributes.WEIGHTS_0)
        cDJoints = checkData(gltf, gltf.meshes[count].primitives[0].attributes.JOINTS_0)
        for i in range(len(cDWeigths)):
            for x in cDWeigths[i]:
                weights.append(x)
            for x in cDJoints[i]:
                joints.append(x)
        count = count + 1

        # por si acaso, para que la malla tenga normales consistentes
        object_geometry.fix_normals(True)

        object_vlist = tm.rendering.mesh_to_vertexlist(object_geometry)

        n_triangles = len(object_vlist[4][1]) // 3

        # el pipeline a usar dependerá de si el objeto tiene textura
        # OJO: asumimos que si tiene material, tiene textura
        # pero no siempre es así.
        if hasattr(object_geometry.visual, "material") and hasattr(object_geometry.visual.material, "image"):
            mesh["pipeline"] = tex_pipeline
            has_texture = True
        else:
            mesh["pipeline"] = notex_pipeline
            has_texture = False

        # inicializamos los datos en la GPU
        mesh["gpu_data"] = mesh["pipeline"].vertex_list_indexed(
            n_triangles, GL.GL_TRIANGLES, object_vlist[3]
        )
        print(mesh)
        # copiamos la posición de los vértices
        mesh["gpu_data"].Position[:] = object_vlist[4][1]

        # las normales vienen en vertex_list[5]
        # las manipulamos del mismo modo que los vértices
        mesh["gpu_data"].Normal[:] = object_vlist[5][1]

        mesh["gpu_data"].BoneWeights[:] = weights
        mesh["gpu_data"].BoneIndices[:] = joints
        # con (o sin) textura es diferente el procedimiento
        # aunque siempre en vertex_list[6] viene la información de material
        if has_texture:
            # copiamos la textura
            # trimesh ya la cargó, solo debemos copiarla a la GPU
            # si no se usa trimesh, el proceso es el mismo,
            # pero se debe usar Pillow para cargar la imagen
            mesh["texture"] = texture_2D_setup(object_geometry.visual.material.image)
            # copiamos las coordenadas de textura en el parámetro uv
            mesh["gpu_data"].uv[:] = object_vlist[6][1]
        else:
            # usualmente el color viene como c4B/static en vlist[6][0], lo que significa "color de 4 bytes". idealmente eso debe verificarse
            mesh["gpu_data"].Color[:] = object_vlist[6][1]

        vertex_lists[object_id] = mesh


    # instanciamos nuestra Arcball
    arcball = Arcball(
        np.identity(4),
        np.array((width, height), dtype=float),
        1.5,
        np.array([0.0, 0.0, 0.0]),
    )

    @window.event
    def on_mouse_press(x, y, button, modifiers):
        print("press", x, y, button, modifiers)
        arcball.down((x, y))

    @window.event
    def on_mouse_release(x, y, button, modifiers):
        print("release", x, y, button, modifiers)
        print(arcball.pose)

    @window.event
    def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
        print("drag", x, y, dx, dy, buttons, modifiers)
        arcball.drag((x, y))

    @window.event
    def on_draw():
        GL.glClearColor(0.5, 0.5, 0.5, 1.0)
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        GL.glLineWidth(1.0)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)

        window.clear()

        for object_geometry in vertex_lists.values():
            # dibujamos cada una de las mallas con su respectivo pipeline
            pipeline = object_geometry["pipeline"]
            pipeline.use()
            pipeline["transform"] = (tr.translate(-0.3, 0, 0) @ arcball.pose).reshape(16, 1, order="F")
            pipeline["light_position"] = np.array([-1.0, 1.0, -1.0])
            for i in range(70):
                pipeline["bone"+str(i)] = v_bones[i].reshape(16, 1, order="F")

            if "texture" in object_geometry:
                GL.glBindTexture(GL.GL_TEXTURE_2D, object_geometry["texture"])
            else:
                # esto "activa" una textura nula
                GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

            object_geometry["gpu_data"].draw(pyglet.gl.GL_TRIANGLES)

        # hay que recorrerlo desde un nodo raíz, que almacenamos como atributo del grafo
        root_key = graph[rootJointNode]
        # tenemos que hacer un recorrido basado en profundidad (DFS).
        # networkx provee una función que nos entrega dicho recorrido!
        edges = list(nx.edge_dfs(graph, source="root"))

        # a medida que nos movemos por las aristas vamos a necesitar la transformación de cada nodo
        # partimos con la transformación del nodo raíz
        transformations = {rootJointNode: graph.nodes[rootJointNode]["transform"] @ tr.uniformScale(node_scale)  @ tr.translate(node_scale, 0, 0)  @ arcball.pose}
        
        axis_pipeline["view"] = tr.lookAt(
            np.array([0, 0, 5]), np.array([0, 0, 0]), np.array([0, 1, 0])
        ).reshape(16, 1, order="F")
        axis_pipeline["projection"] = tr.perspective(45, float(width) / float(height), 0.1, 100).reshape(
            16, 1, order="F"
        )
        for src, dst in edges:
            current_node = graph.nodes[dst]

            if not dst in transformations:
                dst_transform = current_node["transform"]
                transformations[dst] = transformations[src] @ dst_transform

            if "mesh" in current_node:
                
                
                current_pipeline = axis_pipeline
                current_pipeline.use()
                current_pipeline["transform"] = (transformations[dst]).reshape(16, 1, order="F")

                for attr in current_node.keys():
                    if attr in ("mesh", "pipeline", "transform", "mode"):
                        continue

                    current_attr = current_node[attr]
                    current_size = current_node[attr].shape[0]

                    if len(current_node[attr].shape) > 1:
                        current_size = current_size * current_node[attr].shape[1]
                        
                    current_pipeline[attr] = current_node[attr].reshape(
                        current_size, 1, order="F"
                    )

                draw_mode = current_node.get("mode", GL.GL_LINE)
                current_node["mesh"].draw(draw_mode)


    pyglet.app.run(1/60.0)

import os.path
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pyglet
import pyglet.gl as GL
import trimesh as tm

import struct
from pygltflib import GLTF2

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname((os.path.abspath(__file__)))))
)

import grafica.transformations as tr
from grafica.utils import load_pipeline


# esta función crea nuestro grafo de escena.
# esencialmente hace lo siguiente:
# sol -> tierra -> luna
# pero cada elemento tiene a su vez dos nodos: geometry (su modelo 3d) y axis (su eje de coordenadas)
#
# lo hacemos todo con la biblioteca networkx.
# cosas como el pipeline correspondiente a cada malla y los atributos que reciben los pipelines
# son almacenadas como atributos de cada nodo de la red.
def create_solar_system(
    mesh, mesh_pipeline, axis, axis_pipeline, bones_pos, nodes
):
    for i, node in enumerate(nodes):
        print(i, node)

    graph = nx.DiGraph(root="root")
    graph.add_node('root')

    for i, node in enumerate(nodes):
        if node.name in ('Camera', 'Light'):
            continue

        node_mesh = None
        node_pipeline = None
        if node.mesh is None and node.name != 'Armature':
            node_mesh = axis
            node_pipeline = axis_pipeline
        elif node.mesh is not None:
            node_mesh = mesh
            node_pipeline = mesh_pipeline

        graph.add_node(
            node.name,
            mesh=node_mesh,
            pipeline=node_pipeline,
            transform=bones_pos.get(i, tr.identity()),
            mode=GL.GL_LINES if node.mesh is None else GL.GL_TRIANGLES,
            color=np.array((1.0, 0.73, 0.03)) if not node.mesh is None else None
        )
        print("node added", node.name)
        for dst in node.children:
            graph.add_edge(node.name, nodes[dst].name)
            print(node.name, '->', nodes[dst].name)

    print(graph)
    graph.add_edge('root', 'Armature')

    return graph


# esta función actualiza el grafo de escena en función del tiempo
# en este caso, hace algo similar a lo que hemos hecho en ejemplos anteriores
# al asignar rotaciones que dependen del tiempo transcurrido en el programa
def update_solar_system(dt, window):
    window.program_state["total_time"] += dt
    total_time = window.program_state["total_time"]

    graph = window.program_state["scene_graph"]

    graph.nodes[graph.graph["root"]]["transform"] = tr.rotationY(total_time)

    # para acceder a un nodo del grafo utilizamos su atributo .nodes
    # cada nodo es almacenado como un diccionario
    # por tanto, accedemos a él y a sus atributos con llaves de diccionario
    # que conocemos porque nosotres construimos el grafo


if __name__ == "__main__":
    width = 960
    height = 960

    window = pyglet.window.Window(width, height)

    filename = "assets/cube2.gltf"
    gltf = GLTF2().load(filename)
    data = gltf.get_data_from_buffer_uri(gltf.buffers[0].uri)
    bones_num = gltf.skins[0].joints

    inverse_accesor = gltf.skins[0].inverseBindMatrices
    inverse_bind = []
    print("Inverse Bind: ")
    for i in range(gltf.accessors[inverse_accesor].count):
        index = (
            gltf.bufferViews[inverse_accesor].byteOffset
            + gltf.accessors[inverse_accesor].byteOffset
            + i * 64
        )  # the location in the buffer of this vertex
        d = data[index : index + 64]  # the vertex data
        v = struct.unpack("<ffffffffffffffff", d)  # convert from base64 to three floats

        inverse_bind.append(np.array(v).reshape(4, 4))
        print(i, np.array(v).reshape(4, 4, order='F'))

    bones_pos = {}
    for i in range(len(bones_num)):
        node = bones_num[i]
        rotation = tr.identity()
        translation = tr.identity()
        scale = tr.identity()
        if gltf.nodes[node].rotation:
            rotation = tm.transformations.quaternion_matrix(gltf.nodes[node].rotation)
            print("rotation:")
            print(rotation)
        if gltf.nodes[node].translation:
            translation = tm.transformations.translation_matrix(
                gltf.nodes[node].translation
            )
            print("translation:")
            print(scale)
        if gltf.nodes[node].scale:
            scale = tr.scale(*gltf.nodes[node].scale)
            print("scale:")
            print(scale)
        bones_pos[node] = translation @ rotation @ scale

    print('bones_pos', bones_pos)

    # cargamos una esfera y la convertimos en una bola de diámetro 1
    mesh = tm.load("assets/cube2.gltf", force="mesh")
    print(mesh)
    #model_scale = tr.uniformScale(1)
    #model_translate = tr.translate(*-mesh.centroid)
    #mesh.apply_transform(model_scale @ model_translate)

    solar_pipeline = load_pipeline(Path(os.path.dirname(__file__)) / "mesh_vertex_program.glsl", Path(os.path.dirname(__file__)) / ".." / "hello_world" / "fragment_program.glsl")

    mesh_vertex_list = tm.rendering.mesh_to_vertexlist(mesh)
    mesh_gpu = solar_pipeline.vertex_list_indexed(
        len(mesh_vertex_list[4][1]) // 3, GL.GL_TRIANGLES, mesh_vertex_list[3]
    )
    mesh_gpu.position[:] = mesh_vertex_list[4][1]

    # creamos los ejes. los graficaremos con GL_LINES
    axis_positions = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1])
    axis_colors = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1])
    axis_indices = np.array([0, 1, 2, 3, 4, 5])

    axis_pipeline = load_pipeline(Path(os.path.dirname(__file__)) / "line_vertex_program.glsl", Path(os.path.dirname(__file__)) / ".." / "hello_world" / "fragment_program.glsl")

    axis_gpu = axis_pipeline.vertex_list_indexed(6, GL.GL_LINES, axis_indices)
    axis_gpu.position[:] = axis_positions
    axis_gpu.color[:] = axis_colors

    # creamos el grafo de escena con la función definida más arriba
    graph = create_solar_system(
        mesh_gpu,
        solar_pipeline,
        axis_gpu,
        axis_pipeline,
        bones_pos,
        gltf.nodes,
    )

    # el estado del programa almacena el grafo de escena en vez de los modelos 3D
    window.program_state = {
        "scene_graph": graph,
        "total_time": 0.0,
        "view": tr.lookAt(
            np.array([5, 5, 5]), np.array([0, 0, 0]), np.array([0, 1, 0])
        ),
        "projection": tr.perspective(45, float(width) / float(height), 0.1, 100),
    }

    @window.event
    def on_draw():
        GL.glClearColor(0.1, 0.1, 0.1, 1.0)
        GL.glLineWidth(2.0)
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
        GL.glEnable(GL.GL_DEPTH_TEST)

        window.clear()

        # configuramos la vista y proyección de los pipelines
        # estas no cambian durante el programa. de hecho pudimos hacerlo antes
        solar_pipeline.use()

        solar_pipeline["view"] = window.program_state["view"].reshape(16, 1, order="F")
        solar_pipeline["projection"] = window.program_state["projection"].reshape(
            16, 1, order="F"
        )

        axis_pipeline.use()

        axis_pipeline["view"] = window.program_state["view"].reshape(16, 1, order="F")
        axis_pipeline["projection"] = window.program_state["projection"].reshape(
            16, 1, order="F"
        )

        # ahora procederemos a dibujar nuestro grafo de escena.
        graph = window.program_state["scene_graph"]

        # hay que recorrerlo desde un nodo raíz, que almacenamos como atributo del grafo
        root_key = graph.graph["root"]
        # tenemos que hacer un recorrido basado en profundidad (DFS).
        # networkx provee una función que nos entrega dicho recorrido!
        edges = list(nx.edge_dfs(graph, source=root_key))

        # a medida que nos movemos por las aristas vamos a necesitar la transformación de cada nodo
        # partimos con la transformación del nodo raíz
        transformations = {root_key: graph.nodes[root_key].get("transform", tr.identity())}

        for src, dst in edges:
            current_node = graph.nodes[dst]

            if not dst in transformations:
                dst_transform = current_node.get("transform", tr.identity())
                transformations[dst] = transformations[src] @ dst_transform

            if "mesh" in current_node and current_node['mesh'] is not None:
                current_pipeline = current_node["pipeline"]
                current_pipeline.use()

                current_pipeline["transform"] = transformations[dst].reshape(
                    16, 1, order="F"
                )

                for attr in current_node.keys():
                    if attr in ("mesh", "pipeline", "transform", "mode"):
                        continue

                    if current_node[attr] is None:
                        continue

                    current_attr = current_node[attr]
                    current_size = current_node[attr].shape[0]

                    if len(current_node[attr].shape) > 1:
                        current_size = current_size * current_node[attr].shape[1]

                    current_pipeline[attr] = current_node[attr].reshape(
                        current_size, 1, order="F"
                    )

                draw_mode = current_node.get("mode", GL.GL_TRIANGLES)
                current_node["mesh"].draw(draw_mode)

    pyglet.clock.schedule_interval(update_solar_system, 1 / 60.0, window)
    pyglet.app.run(1 / 60.0)

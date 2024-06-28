import pyglet
import pyglet.gl as GL
import trimesh as tm
import networkx as nx
import numpy as np
import os
from pathlib import Path
import sys

import struct
from pygltflib import GLTF2

if sys.path[0] != "":
    sys.path.insert(0, "")

# una función auxiliar para cargar shaders
import grafica.transformations as tr
from grafica.utils import load_pipeline

from grafica.arcball import Arcball
from grafica.textures import texture_2D_setup

from examples.rigging.app2 import create_axis_graph

if __name__ == "__main__":
    width = 400
    height = 400
    window = pyglet.window.Window(width, height)

    # dependiendo de lo que contenga el archivo a cargar,
    # trimesh puede entregar una malla (mesh)
    # o una escena (compuesta de mallas)
    # con esto forzamos que siempre entregue una escena
    asset = tm.load(sys.argv[1], force="scene")
    mesh = tm.load(sys.argv[1], force="mesh")
    print("Camera", asset.camera_transform)
    print("Graph", asset.graph.to_edgelist())
    a = asset.camera_transform
    print(a[2][3])
    #for x in digraph.adjacency():
    #    print(x)

    print("mesh scale", mesh.scale)
    print(type(asset))
    factor = 2
    # de acuerdo a la documentación de trimesh, esto centra la escena
    # no es igual a trabajar con una malla directamente
    scale = 1 / (asset.scale)
    print("scale", asset.scale)
    print("primera escala", asset.scale)
    
    first = asset.scale
    # y esto la escala. se crea una copia, por eso la asignación
    asset = asset.scaled(scale)
    print("segunda escala", asset.scale)
    second = asset.scale
    # como no todos los archivos que carguemos tendrán textura,
    # tendremos dos pipelines
    tex_pipeline = load_pipeline(
        Path(os.path.dirname(__file__)) / "vertex_program.glsl",
        Path(os.path.dirname(__file__)) / "fragment_program.glsl",
    )

    notex_pipeline = load_pipeline(
        Path(os.path.dirname(__file__)) / "vertex_program_notex.glsl",
        Path(os.path.dirname(__file__)) / "fragment_program_notex.glsl",
    )

    # aquí guardaremos las mallas del modelo que graficaremos
    vertex_lists = {}

    gltf = GLTF2().load(sys.argv[1])

    # con esto iteramos sobre las mallas
    for object_id, object_geometry in asset.geometry.items():
        mesh = {}

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

        # copiamos la posición de los vértices
        mesh["gpu_data"].position[:] = object_vlist[4][1]

        # las normales vienen en vertex_list[5]
        # las manipulamos del mismo modo que los vértices
        mesh["gpu_data"].normal[:] = object_vlist[5][1]

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
            mesh["gpu_data"].color[:] = object_vlist[6][1]

        vertex_lists[object_id] = mesh

     # creamos los ejes. los graficaremos con GL_LINES
    axis_positions = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1])
    axis_colors = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1])
    axis_indices = np.array([0, 1, 2, 3, 4, 5])

    axis_pipeline = load_pipeline(Path(os.path.dirname(__file__)) / "line_vertex_program.glsl", Path(os.path.dirname(__file__)) / ".." / "hello_world" / "fragment_program.glsl")

    axis_gpu = axis_pipeline.vertex_list_indexed(6, GL.GL_LINES, axis_indices)
    axis_gpu.position[:] = axis_positions
    axis_gpu.color[:] = axis_colors

    # creamos el grafo de escena con la función definida más arriba
    graph = create_axis_graph(
        gltf,
        axis_gpu,
        axis_pipeline,
        1,
    )

    # el estado del programa almacena el grafo de escena en vez de los modelos 3D
    window.program_state = {
        "scene_graph": graph,
        "total_time": 0.0,
        "view": tr.lookAt(
            np.array([0, 0, scale]), np.array([0, 0, 0]), np.array([0, 1, 0])
        ),
        "projection": tr.perspective(45, float(width) / float(height), 0.1, 100),
    }

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
    def on_mouse_scroll(x, y, scroll_x, scroll_y):
        print("scroll", x, y, scroll_x, scroll_y)
        arcball.scroll((scroll_y))

    @window.event
    def on_draw():
        GL.glClearColor(0.5, 0.5, 0.5, 1.0)
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        GL.glLineWidth(1.0)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)

        window.clear()

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


        for object_geometry in vertex_lists.values():
            # dibujamos cada una de las mallas con su respectivo pipeline
            pipeline = object_geometry["pipeline"]
            pipeline.use()

            pipeline["transform"] = arcball.pose.reshape(16, 1, order="F")
            pipeline["light_position"] = np.array([-1.0, 1.0, -1.0])

            if "texture" in object_geometry:
                GL.glBindTexture(GL.GL_TEXTURE_2D, object_geometry["texture"])
            else:
                # esto "activa" una textura nula
                GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

            object_geometry["gpu_data"].draw(pyglet.gl.GL_TRIANGLES)

    pyglet.app.run()

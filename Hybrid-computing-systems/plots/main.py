import json
import plotly.graph_objs as go
import plotly.offline as pyo
import os
import sys

objs_GPUFull = list()
objs_GPUCore = list()
objs_CPU = list()


def read_file() -> list:
    file_path = os.path.join(os.path.dirname(__file__), 'benchmarks_data', 'results.json')
    
    with open(file_path, "r") as file:
        data = json.load(file)
        return data["benchmarks"]


def plot():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[obj['number_elements'] for obj in objs_CPU],
                             y=[obj['real_time'] for obj in objs_CPU], name="CPU"))
    fig.add_trace(go.Scatter(x=[obj['number_elements'] for obj in objs_GPUFull],
                             y=[obj['real_time'] for obj in objs_GPUFull], name="GPUFull"))
    fig.add_trace(go.Scatter(x=[obj['number_elements'] for obj in objs_GPUCore],
                             y=[obj['real_time'] for obj in objs_GPUCore], name="GPUCore"))
    fig.update_layout(legend_orientation="h",
                  legend=dict(x=.5, xanchor="center"),
                  title="Зависимость времяни выполнения от количества элементов в массиве",
                  xaxis_title="Количество элементов в массиве",
                  yaxis_title="Время выполнения(ns)",
                  margin=dict(l=0, r=0, t=30, b=0))

    pyo.plot(fig, filename=path_output_file(), auto_open=True)


def path_output_file():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(current_dir, "benchmark_plot.html")
    return output_file


def parsing_json(list_obj):
    for obj in list_obj:
        if "CPU" in obj["name"]:
            obj_CPU = {"name": (obj["name"].split("/"))[0], "number_elements": (obj["name"].split("/"))[1], "real_time": obj["real_time"]}
            objs_CPU.append(obj_CPU)
        elif "GPUFull" in obj["name"]:
            obj_GPU = {"name": (obj["name"].split("/"))[0], "number_elements": (obj["name"].split("/"))[1], "real_time": obj["real_time"]}
            objs_GPUFull.append(obj_GPU)
        elif "GPUCore" in obj["name"]:
            obj_GPU = {"name": (obj["name"].split("/"))[0], "number_elements": (obj["name"].split("/"))[1], "real_time": obj["real_time"]}
            objs_GPUCore.append(obj_GPU)


if __name__ == '__main__':
    list_obj = read_file()
    parsing_json(list_obj)
    plot()
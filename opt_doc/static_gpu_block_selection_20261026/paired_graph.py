import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import taichi_forge as ti

p=argparse.ArgumentParser()
p.add_argument('--arch', required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
ti.init(arch=getattr(ti,a.arch),enable_fallback=False,offline_cache=False,
        compile_tier='fast',advanced_optimization=False)
results=[]
for name,count,rounds in [('pointwise',1024,0),('compute',4096,48),('large',65536,0)]:
    x=ti.ndarray(ti.f32,count)
    y=ti.ndarray(ti.f32,count)
    x.from_numpy(np.linspace(0,1,count,dtype=np.float32))
    graphs=[]
    blocks=[]
    cold=[]
    images=[]
    for block in (128,0):
        @ti.kernel
        def work(src:ti.types.ndarray(ti.f32,1),dst:ti.types.ndarray(ti.f32,1)):
            ti.loop_config(block_dim=block)
            for i in range(count):
                v=src[i]
                for j in range(rounds):
                    v=ti.sqrt(v+2.0)+0.01*(j%3)
                dst[i]=v*2.0+1.0
        start=perf_counter()
        work(x,y)
        ti.sync()
        cold.append(perf_counter()-start)
        images.append(y.to_numpy())
        blocks.append(work.task_manifest(x,y)[0].selected_block_size)
        builder=ti.graph.GraphBuilder()
        for _ in range(64):
            builder.dispatch(work,ti.graph.Arg(ti.graph.ArgKind.NDARRAY,'src',ti.f32,ndim=1),
                             ti.graph.Arg(ti.graph.ArgKind.NDARRAY,'dst',ti.f32,ndim=1))
        graph=builder.compile()
        graph.prepare_telemetry(True)
        graphs.append(graph)
    np.testing.assert_array_equal(images[0],images[1])
    arguments=dict(src=x,dst=y)
    for _ in range(8):
        for graph in graphs:
            ticket=graph.submit(arguments,telemetry=True)
            report=ticket.telemetry()
            assert report.gpu_duration_ns is not None, report
    samples=[[],[]]
    host=[[],[]]
    for i in range(31):
        for variant in ((0,1) if i%2 else (1,0)):
            t=perf_counter()
            ticket=graphs[variant].submit(arguments,telemetry=True)
            report=ticket.telemetry()
            host[variant].append((perf_counter()-t)*1e6/64)
            samples[variant].append(report.gpu_duration_ns/1e3/64)
    results.append(dict(name=name,blocks=blocks,compile_load_s=cold,
                        gpu_us=[float(np.median(s)) for s in samples],gpu_samples_us=samples,
                        host_us=[float(np.median(s)) for s in host],host_samples_us=host))
    for graph in graphs: graph.close()
data=dict(arch=a.arch,scope='same process, explicit 128 vs automatic; 64-node graph GPU interval / 64; 31 alternating samples',results=results)
a.output.write_text(json.dumps(data,indent=2))
print(json.dumps(data))

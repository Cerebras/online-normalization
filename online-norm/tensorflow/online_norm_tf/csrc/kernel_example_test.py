import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
from tensorflow.core.protobuf import config_pb2
import numpy as np
try:
    zero_out_module = tf.load_op_library('./online_norm_gpu.so')
    modes = ["cpu", "gpu"]
    # modes = ["gpu"]
except:
    zero_out_module = tf.load_op_library('./online_norm_cpu.so')
    modes = ["cpu"]


dtypes = [tf.float32, tf.float16]
functions = [
    zero_out_module.online_norm_fwd,
    zero_out_module.online_norm_bwd,
]

tf.logging.set_verbosity(tf.logging.INFO)
tf.debugging.set_log_device_placement(True)
config = config_pb2.ConfigProto()
config.allow_soft_placement = False
# Don't perform optimizations for tests so we don't inadvertently run
# gpu ops on cpu
config.graph_options.optimizer_options.opt_level = -1


# data:
N = 4
C = 3
D = 5
data_N_C_D = np.random.uniform(size=[N, C, D])
data_N_C = np.random.uniform(size=[N, C])
data_C = np.random.uniform(size=[C])


for funcs in functions:
    for dtype in dtypes:
        results_dict = {}
        for mode in modes:
            with tf.Session(config=config) as sess:
                with tf.device(f"{mode}:0"):
                    if funcs == zero_out_module.online_norm_fwd:
                        np_input = np.ones((N,C))
                        inputs = tf.constant(np_input, dtype=dtype)
                        result = funcs(
                            mu=tf.cast(inputs, tf.float32),
                            var=tf.cast(inputs, tf.float32),
                            in_s_mu=tf.cast(inputs[0,:], tf.float32),
                            in_s_var=tf.cast(inputs[0,:], tf.float32),
                            afwd=0.3,
                            eps=0.4,
                            T=dtype
                        )
                    elif funcs == zero_out_module.online_norm_bwd:
                        np_input = np.ones((N,C,D))
                        inputs = tf.constant(np_input, dtype=dtype)
                        result = funcs(
                            grad_out=inputs,
                            in_v=tf.cast(inputs[0,:,0], tf.float32),
                            in_u=tf.cast(inputs[0,:,0], tf.float32),
                            out=inputs,
                            scale=inputs[:,:,0],
                            abkw=0.3,
                        )
                    tf.logging.info(f"function: {funcs}, Device: {mode}, dtype: {dtype}")
                    ans = sess.run(result)
                    # tf.logging.info(f"found ans: \n {ans}")

            for i, val in enumerate(ans):
                mode_dict = results_dict.get(i, {})
                mode_dict[mode] = val
                results_dict[i] = mode_dict
        tf.logging.info(f"function: {funcs}, dtype: {dtype}")
        for k, v in results_dict.items():
            mode_dict = v
            if len(modes) == 2:
                error=np.sum(abs(v["cpu"] - v["gpu"]))
                tf.logging.info(f"{k}: error:{error}")
            else:
                error=v[modes[0]]
                tf.logging.info(f"{k}: output:{error}")
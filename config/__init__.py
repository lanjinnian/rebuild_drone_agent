#关键帧选取
KEY_FRAME_DELETE_RATIO = 0.7

#分块大小
CHUNK_SIZE = 4
CHUNK_OVERLAP_SIZE = 2

#拼接相关
ALIGN_CONF_THRESHOLD = 0.0      #用于拼接计算的置信度阈值
ALIGN_IRLS_DELTA = 0.1          #鲁棒估计减低某点权重的阈值
ALIGN_IRLS_MAX_ITERS = 5        #鲁棒估计迭代次数
ALIGN_MIN_POINTS = 1            #拼接对齐所需要最少有效点次数

#输出地址
RESULT_DIR = "result"

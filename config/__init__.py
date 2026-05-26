#关键帧选取
KEY_FRAME_DELETE_RATIO = 0.7

#分块大小
CHUNK_SIZE = 20
CHUNK_OVERLAP_SIZE = 10

#读取类型，可选：video、image
READ_TYPE_VIDEO = "video"
READ_TYPE_IMAGE = "image"
READ_TYPE = READ_TYPE_IMAGE
READ_TYPE_CHOICES = (READ_TYPE_VIDEO, READ_TYPE_IMAGE)

#视频读取参数
VIDEO_SEGMENT_SECONDS = 30
VIDEO_EXTRACT_FPS = 6

#图片读取参数
IMAGE_SEGMENT_SIZE = 200
IMAGE_SAMPLE_INTERVAL = 3

#去除遮罩
REMOVE_MASK_ENABLE = True
REMOVE_MASK_MODEL_PATH = "models/model_final_v2_city.pth"
REMOVE_MASK_MODEL_URL = "https://github.com/CoinCheung/BiSeNet/releases/download/0.0.0/model_final_v2_city.pth"
REMOVE_MASK_SOURCE_PATH = "models/bisenetv2.py"
REMOVE_MASK_SOURCE_URL = "https://raw.githubusercontent.com/CoinCheung/BiSeNet/master/lib/models/bisenetv2.py"
REMOVE_MASK_DEVICE = None
REMOVE_MASK_CLASS_IDS = tuple(range(10, 19))
REMOVE_MASK_MEAN = (0.485, 0.456, 0.406)
REMOVE_MASK_STD = (0.229, 0.224, 0.225)

#拼接相关
ALIGN_CONF_THRESHOLD = 0.3      #用于拼接计算的置信度阈值
ALIGN_IRLS_DELTA = 0.1          #鲁棒估计减低某点权重的阈值
ALIGN_IRLS_MAX_ITERS = 5        #鲁棒估计迭代次数
ALIGN_MIN_POINTS = 1            #拼接对齐所需要最少有效点次数
ALIGN_LOOP_ENABLE = True        #是否启用非相邻分块回环约束
ALIGN_LOOP_MIN_CHUNK_GAP = 2    #回环约束要求两个分块至少间隔几个分块
ALIGN_LOOP_MAX_PAIRS = 8        #最多使用多少组回环约束
ALIGN_LOOP_MIN_MATCHES = 20     #回环帧对估计Sim3所需最少特征匹配点数
ALIGN_LOOP_MIN_SCORE = 0.08     #图像特征重叠分数阈值
ALIGN_LOOP_TOP_K = 2            #每个分块最多保留多少个候选回环
ALIGN_LOOP_OPT_MAX_ITERS = 50   #Sim3图优化最大迭代次数
ALIGN_LOOP_WINDOW_RADIUS = 5    #回环共同推理时，以匹配帧为中心向两侧取多少帧
ALIGN_LOOP_MODEL_NAME_OR_PATH = "facebook/VGGT-1B"
ALIGN_GLB_CONF_THRES = 60.0     #最终GLB输出的置信度百分位阈值

#输出地址
RESULT_DIR = "data/result"

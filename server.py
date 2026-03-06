from dotenv import load_dotenv
import os
import sys
from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI, File, UploadFile, HTTPException, Header
from fastapi.responses import HTMLResponse
import uvicorn
import numpy as np
import cv2
import asyncio
import logging
# from paddleocr import PaddleOCR
import torch
from PIL import Image, ImageFile
from io import BytesIO
from pydantic import BaseModel
from paddleocr import PaddleOCR  # 菴ｿ逕ｨPaddleOCR GPU迚域悽
# import cn_clip.clip as clip  # 豕ｨ驥頑脂蜴滓怏逧�lip蟇ｼ蜈･
ImageFile.LOAD_TRUNCATED_IMAGES = True

on_linux = sys.platform.startswith('linux')

# 蠢�｡ｻ蝨ｨ蟇ｼ蜈･immich_adapter荵句燕蜉霓ｽ.env譁�ｻｶ
load_dotenv()
from immich_adapter import immich_adapter  # 菴ｿ逕ｨimmich騾る�蝎?

# 驟咲ｽｮ譌･蠢
logger = logging.getLogger("uvicorn")
# 隶ｾ鄂ｮ譌･蠢礼ｺｧ蛻ｫ
log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_str, logging.INFO)
logger.setLevel(log_level)

api_auth_key = os.getenv("API_AUTH_KEY", "mt_photos_ai_extra")
http_port = int(os.getenv("HTTP_PORT", "8060"))
server_restart_time = int(os.getenv("SERVER_RESTART_TIME", "300"))
env_auto_load_txt_modal = os.getenv("AUTO_LOAD_TXT_MODAL", "off") == "on" # 譏ｯ蜷ｦ閾ｪ蜉ｨ蜉霓ｽCLIP譁�悽讓｡蝙具ｼ悟ｼ蜷ｯ蜿ｯ莉･莨伜喧隨ｬ荳谺｡謳懃ｴ｢譌ｶ逧�桃蠎秘溷ｺｦ,譁�悽讓｡蝙句頃逕ｨ700螟嗄蜀�ｭ

# clip_model_name = os.getenv("CLIP_MODEL")  # 遘ｻ蛻ｰimmich_adapter荳ｭ邂｡逅?


ocr_model = None
# clip_processor = None  # 荳榊�髴隕�ｼ御ｽｿ逕ｨimmich騾る�蝎?
# clip_model = None  # 荳榊�髴隕�ｼ御ｽｿ逕ｨimmich騾る�蝎?

restart_task = None
restart_lock = asyncio.Lock()

device = "cuda" if torch.cuda.is_available() else "cpu"

class ClipTxtRequest(BaseModel):
    text: str

def load_ocr_model():
    """鬚�刈霓ｽOCR讓｡蝙"""
    global ocr_model
    if ocr_model is None:
        logger.info("Loading OCR model 'PaddleOCR' to memory")
        
        # 譬ｹ謐ｮ PaddleOCR 3.0 螳俶婿譁�｡｣�御ｽｿ逕ｨ鮟倩ｮ､驟咲ｽ?
        # 鮟倩ｮ､菴ｿ逕ｨ PP-OCRv5_server 讓｡蝙具ｼ梧髪謖∽ｸｭ闍ｱ譁�ｯ�悪
        # PaddleOCR 莨夊�蜉ｨ荳玖ｽｽ讓｡蝙句芦邉ｻ扈滄ｻ倩ｮ､郛灘ｭ倡岼蠖
        # 豕ｨ諢擾ｼ壼宵譛牙惠蟾ｲ譛牙ｮ梧紛讓｡蝙区枚莉ｶ譌ｶ謇崎�菴ｿ逕?text_detection_model_dir 蜿よ焚
        ocr_model = PaddleOCR()
        if torch.cuda.is_available():
            logger.info("PaddleOCR initialized with GPU acceleration")
        else:
            logger.info("PaddleOCR initialized with CPU")
        logger.info("PaddleOCR models will be automatically cached by the system")
        # https://paddlepaddle.github.io/PaddleOCR/main/en/quick_start.html

def load_clip_model():
    """鬚�刈霓ｽCLIP讓｡蝙 - 菴ｿ逕ｨimmich騾る�蝎?"""
    try:
        # 鬚�刈霓ｽ隗�ｧ牙柱譁�悽讓｡蝙
        immich_adapter.load_clip_visual_model()
        if env_auto_load_txt_modal:
            immich_adapter.load_clip_textual_model()
    except Exception as e:
        print(f"Error loading CLIP models: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 蜷ｯ蜉ｨ莠倶ｻｶ
    import onnxruntime as ort
    logger.info("Using PaddleOCR with GPU support")
    logger.info(f"LOG_LEVEL: {os.getenv('LOG_LEVEL', 'ERROR')}")
    logger.info(f"MODEL_TTL: {os.getenv('MODEL_TTL', '0')}")
    logger.info(f"FACE_MODEL_NAME: {immich_adapter.face_model_name}")
    logger.info(f"CLIP_MODEL_NAME: {immich_adapter.clip_model_name}")
    logger.info(f"FACE_THRESHOLD: {immich_adapter.face_threshold}")
    logger.info(f"DEVICE: {device}")
    logger.info(f"CUDA_AVAILABLE: {torch.cuda.is_available()}")
    # 霎灘�ONNX Runtime謇ｧ陦梧署萓帷ｨ句ｺ丈ｿ｡諱ｯ
    available_providers = ort.get_available_providers()

    logger.info(f"ONNX_PROVIDERS: {available_providers}")
    # 譽譟･CUDA謇ｧ陦梧署萓帷ｨ句ｺ
    if 'CUDAExecutionProvider' in available_providers:
        logger.info(f"CUDA_RUNTIME: Available")
    else:
        logger.info(f"CUDA_RUNTIME: Not Available")
    if env_auto_load_txt_modal:
        load_clip_model()
        logger.info("Auto-loaded CLIP text model")
    
    yield
    
    # 蜈ｳ髣ｭ莠倶ｻｶ
    if restart_task and not restart_task.done():
        restart_task.cancel()
        try:
            await restart_task
        except asyncio.CancelledError:
            pass

# 蛻帛ｻｺ FastAPI 蠎皮畑螳樔ｾ
app = FastAPI(lifespan=lifespan)

async def restart_timer():
    await asyncio.sleep(server_restart_time)
    restart_program()

@app.middleware("http")
async def activity_monitor(request, call_next):
    global restart_task

    async with restart_lock:
        if restart_task and not restart_task.done():
            restart_task.cancel()

        restart_task = asyncio.create_task(restart_timer())

    response = await call_next(request)
    return response


async def verify_header(api_key: str = Header(...)):
    # 蝨ｨ霑咎㈹郛門�鬪瑚ｯ�ｻ霎托ｼ御ｾ句ｦよ｣譟?api_key 譏ｯ蜷ｦ譛画譜
    if api_key != api_auth_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key


def to_fixed(num):
    return str(round(num, 2))

def convert_paddleocr_to_json(paddleocr_output):
    """
    蟆?PaddleOCR 3.0 逧�ｾ灘�霓ｬ謐｢荳ｺ JSON 譬ｼ蠑
    
    Args:
        paddleocr_output: PaddleOCR 3.0 逧�次蟋玖ｾ灘?
        
    Returns:
        dict: 蛹�性 texts, scores, boxes 逧�ｭ怜?
    """
    logger.debug(f"convert_paddleocr_to_json input type: {type(paddleocr_output)}")
    logger.debug(f"convert_paddleocr_to_json input content: {paddleocr_output}")
    
    texts = []
    scores = []
    boxes = []
    
    try:
        # PaddleOCR 3.0 譁ｰ譬ｼ蠑擾ｼ夊ｿ泌屓蟄怜�蛹�性 rec_texts, rec_scores, rec_polys 遲牙ｭ玲ｮ?
        if isinstance(paddleocr_output, list) and len(paddleocr_output) > 0:
            # 譽譟･譏ｯ蜷ｦ荳ｺ譁ｰ逧�ｭ怜�譬ｼ蠑
            if isinstance(paddleocr_output[0], dict):
                result_dict = paddleocr_output[0]
                
                # 謠仙叙譁�悽
                if 'rec_texts' in result_dict:
                    texts = [str(text) for text in result_dict['rec_texts']]
                
                # 謠仙叙鄂ｮ菫｡蠎?
                if 'rec_scores' in result_dict:
                    scores = [f"{float(score):.2f}" for score in result_dict['rec_scores']]
                
                # 謠仙叙霎ｹ逡梧｡�攝譬?
                if 'rec_polys' in result_dict:
                    for poly in result_dict['rec_polys']:
                        if hasattr(poly, 'tolist'):
                            # 螟�炊 numpy 謨ｰ扈
                            poly = poly.tolist()
                        
                        if isinstance(poly, list) and len(poly) == 4:
                            # 隶｡邂礼洸蠖｢霎ｹ逡梧｡?
                            xs = [point[0] for point in poly]
                            ys = [point[1] for point in poly]
                            
                            x_min, x_max = min(xs), max(xs)
                            y_min, y_max = min(ys), max(ys)
                            
                            width = x_max - x_min
                            height = y_max - y_min
                            
                            boxes.append({
                                'x': to_fixed(x_min),
                                'y': to_fixed(y_min),
                                'width': to_fixed(width),
                                'height': to_fixed(height)
                            })
                
                logger.info(f"PaddleOCR 3.0 new format processed: {len(texts)} texts, {len(scores)} scores, {len(boxes)} boxes")
            
            # 蜈ｼ螳ｹ譌ｧ譬ｼ蠑? [[[x1,y1], [x2,y2], [x3,y3], [x4,y4]], [text, confidence]]
            elif isinstance(paddleocr_output[0], list) and len(paddleocr_output[0]) == 2:
                for line in paddleocr_output:
                    if isinstance(line, list) and len(line) == 2:
                        bbox, text_info = line
                        
                        # 螟�炊霎ｹ逡梧｡�攝譬?
                        if isinstance(bbox, list) and len(bbox) == 4:
                            # 隶｡邂礼洸蠖｢霎ｹ逡梧｡?
                            xs = [point[0] for point in bbox]
                            ys = [point[1] for point in bbox]
                            
                            x_min, x_max = min(xs), max(xs)
                            y_min, y_max = min(ys), max(ys)
                            
                            width = x_max - x_min
                            height = y_max - y_min
                            
                            boxes.append({
                                'x': to_fixed(x_min),
                                'y': to_fixed(y_min),
                                'width': to_fixed(width),
                                'height': to_fixed(height)
                            })
                        
                        # 螟�炊譁�悽蜥檎ｽｮ菫｡蠎ｦ
                        if isinstance(text_info, list) and len(text_info) >= 2:
                            text, confidence = text_info[0], text_info[1]
                            texts.append(str(text))
                            scores.append(f"{float(confidence):.2f}")
                        else:
                            logger.warning(f"Unexpected text_info format: {text_info}")
                            texts.append("")
                            scores.append("0.00")
                    else:
                        logger.warning(f"Unexpected line format: {line}")
                
                logger.info(f"PaddleOCR legacy format processed: {len(texts)} texts, {len(scores)} scores, {len(boxes)} boxes")
            else:
                logger.warning(f"Unexpected paddleocr_output list format: {paddleocr_output[0]}")
        else:
            logger.warning(f"Unexpected paddleocr_output format: {type(paddleocr_output)}")
    
    except Exception as e:
        logger.error(f"Error in convert_paddleocr_to_json: {e}")
        logger.error(f"paddleocr_output: {paddleocr_output}")
    
    logger.debug(f"Conversion result - texts: {len(texts)}, scores: {len(scores)}, boxes: {len(boxes)}")
    
    return {
        'texts': texts,
        'scores': scores,
        'boxes': boxes
    }

@app.get("/", response_class=HTMLResponse)
async def top_info():
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>MT Photos AI Server</title>
    <style>p{text-align: center;}</style>
</head>
<body>
<p style="font-weight: 600;">MT Photos譎ｺ閭ｽ隸�悪譛榊苅</p>
<p>譛榊苅迥ｶ諤�ｼ 霑占｡御ｸ?/p>
<p>菴ｿ逕ｨ譁ｹ豕包ｼ?<a href="https://mtmt.tech/docs/advanced/ocr_api">https://mtmt.tech/docs/advanced/ocr_api</a></p>
</body>
</html>"""
    return html_content


@app.post("/check")
async def check_req(api_key: str = Depends(verify_header)):
    return {
        'result': 'pass',
        "title": "mt-photos-ai譛榊苅 (髮��Immich)",
        "help": "https://mtmt.tech/docs/advanced/ocr_api",
        "device": device,
        "face_model": immich_adapter.face_model_name,
        "clip_model": immich_adapter.clip_model_name,
        "detector_backend": immich_adapter.face_model_name,
        "recognition_model": immich_adapter.face_model_name
    }


@app.post("/restart")
async def check_req(api_key: str = Depends(verify_header)):
    # cuda迚域悽 OCR豐｡譛画仞蟄俶悴驥頑叛髣ｮ鬚假ｼ瑚ｿ呵ｾｹ蜿ｯ莉･蜈ｳ髣ｭ驥榊星
    return {'result': 'unsupported'}
    # restart_program()

@app.post("/restart_v2")
async def check_req(api_key: str = Depends(verify_header)):
    # 鬚�蕗隗ｦ蜿第恪蜉｡驥榊星謗･蜿｣-閾ｪ蜉ｨ驥頑叛蜀�ｭ
    restart_program()
    return {'result': 'pass'}

@app.post("/ocr")
async def process_image(file: UploadFile = File(...), api_key: str = Depends(verify_header)):
    logger.info(f"ocr_process_image Received {file.content_type} file: {file.filename}")
    load_ocr_model()
    image_bytes = await file.read()
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        height, width, _ = img.shape
        if width > 10000 or height > 10000:
            return {'result': [], 'msg': 'height or width out of range'}

        # 譬ｹ謐ｮ PaddleOCR 3.0 螳俶婿譁�｡｣�檎峩謗･隹�?predict 譁ｹ豕
        _result = await asyncio.get_running_loop().run_in_executor(None, ocr_model.predict, img)
        logger.info(f"Raw PaddleOCR result for {file.filename}: {_result}")
        result = convert_paddleocr_to_json(_result)
        del img
        del _result
        logger.info(f"OCR processing completed for {file.filename}, texts found: {len(result.get('texts', []))}, scores: {len(result.get('scores', []))}, boxes: {len(result.get('boxes', []))}")
        if len(result.get('texts', [])) == 0:
            logger.warning(f"No text extracted from {file.filename}, check convert_paddleocr_to_json function")
        return {'result': result}
    except Exception as e:
        logger.error(f"OCR processing error: {e}")
        return {'result': [], 'msg': str(e)}

@app.post("/clip/img")
async def clip_process_image(file: UploadFile = File(...), api_key: str = Depends(verify_header)):
    logger.info(f"clip_process_image Received {file.content_type} file: {file.filename}")
    image_bytes = await file.read()
    try:
        # 菴ｿ逕ｨimmich騾る�蝎ｨ霑幄｡悟崟蜒冗ｼ也?
        result = await asyncio.get_running_loop().run_in_executor(
            None, immich_adapter.encode_image, image_bytes
        )
        logger.info(f"CLIP image processing completed for {file.filename}, result: {result[:3] if len(result) > 3 else result}...")
        return {'result': result}
    except Exception as e:
        logger.error(f"CLIP image processing error: {e}")
        return {'result': [], 'msg': str(e)}

@app.post("/clip/txt")
async def clip_process_txt(request:ClipTxtRequest, api_key: str = Depends(verify_header)):
    logger.info(f"clip_process_text Received text query: {request.text[:50]}...")
    try:
        # 菴ｿ逕ｨimmich騾る�蝎ｨ霑幄｡梧枚譛ｬ郛也?
        result = await asyncio.get_running_loop().run_in_executor(
            None, immich_adapter.encode_text, request.text
        )
        logger.info(f"CLIP text processing completed, result: {result[:3] if len(result) > 3 else result}...")
        return {'result': result}
    except Exception as e:
        logger.error(f"CLIP text processing error: {e}")
        return {'result': [], 'msg': str(e)}

@app.post("/represent")
async def face_represent(file: UploadFile = File(...), api_key: str = Depends(verify_header)):
    """莠ｺ閼ｸ迚ｹ蠕∵署蜿泡PI - 蜈ｼ螳ｹMT-Photos譬ｼ蠑擾ｼ御ｽｿ逕ｨImmich蜷守ｫｯ"""
    logger.info(f"face_represent Received {file.content_type} file: {file.filename}")
    content_type = file.content_type
    image_bytes = await file.read()
    
    try:
        img = None
        if content_type == 'image/gif':
            # 螟�炊GIF譁�ｻｶ逧�ｬｬ荳蟶?
            with Image.open(BytesIO(image_bytes)) as pil_img:
                if pil_img.is_animated:
                    pil_img.seek(0)
                frame = pil_img.convert('RGB')
                np_arr = np.array(frame)
                img = cv2.cvtColor(np_arr, cv2.COLOR_RGB2BGR)
        
        if img is None:
            # 螟�炊蜈ｶ莉門崟蜒乗ｼ蠑
            np_arr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if img is None:
            err = f"The uploaded file {file.filename} is not a valid image format or is corrupted."
            logger.error(err)
            return {'result': [], 'msg': str(err)}
        
        height, width, _ = img.shape
        if width > 10000 or height > 10000:
            return {'result': [], 'msg': 'height or width out of range'}
        
        data = {
            "detector_backend": immich_adapter.face_model_name,
            "recognition_model": immich_adapter.face_model_name
        }
        
        # 菴ｿ逕ｨImmich騾る�蝎ｨ霑幄｡御ｺｺ閼ｸ迚ｹ蠕∵署蜿?
        embedding_objs = await asyncio.get_running_loop().run_in_executor(
            None, _immich_represent, image_bytes
        )
        
        del img
        data["result"] = embedding_objs
        logger.info(f"Face representation completed for {file.filename}, embeddings count: {len(embedding_objs)}")
        return data
        
    except Exception as e:
        if 'set enforce_detection' in str(e):
            return {'result': []}
        logger.error(f"Face representation error: {e}")
        return {'result': [], 'msg': str(e)}

def _immich_represent(image_bytes):
    """菴ｿ逕ｨImmich騾る�蝎ｨ霑幄｡御ｺｺ閼ｸ迚ｹ蠕∵署蜿?"""
    try:
        face_result = immich_adapter.detect_faces(image_bytes)
        # 逶ｴ謗･霑泌屓result謨ｰ扈�ｼ御ｿ晄戟荳札eepFace.represent譬ｼ蠑丞�螳ｹ
        if face_result and 'result' in face_result:
            return face_result['result']
        return []
    except Exception as e:
        logger.error(f"Immich face representation error: {e}")
        return []

async def predict(predict_func, inputs):
    return await asyncio.get_running_loop().run_in_executor(None, predict_func, inputs)


def restart_program():
    print("restart_program")
    python = sys.executable
    os.execl(python, python, *sys.argv)


if __name__ == "__main__":
    uvicorn.run("server:app", host=None, port=http_port)

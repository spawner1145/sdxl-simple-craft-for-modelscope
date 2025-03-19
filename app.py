import subprocess
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def install_dependencies():
    required_packages = [
        "diffusers>=0.24.0",
        "safetensors",
        "transformers>=4.37.0",
        "psutil",
        "gradio"
    ]
    for package in required_packages:
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--upgrade", package],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT
            )
            logger.info(f"Successfully installed/upgraded {package}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install {package}: {e}")
            raise

install_dependencies()

import random
import torch
import os
import psutil
import gc
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    DDIMScheduler,
    HeunDiscreteScheduler,
    KDPM2DiscreteScheduler,
    UniPCMultistepScheduler,
    DEISMultistepScheduler
)
import gradio as gr
from pathlib import Path
import base64
from embeds import get_weighted_text_embeddings_sdxl
from io import BytesIO
from PIL import Image

# 默认参数
DEFAULT_WIDTH = 1064
DEFAULT_HEIGHT = 1600
DEFAULT_STEPS = 30
MIN_WIDTH = 512
MIN_HEIGHT = 512
MIN_STEPS = 20

# 添加内存优化配置
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

os.environ['HF_ENDPOINT'] = 'hf-cdn.sufy.com'

# 打印关键库版本以便调试
def check_versions():
    import diffusers
    import transformers
    logger.info(f"Diffusers version: {diffusers.__version__}")
    logger.info(f"Transformers version: {transformers.__version__}")
    logger.info(f"PyTorch version: {torch.__version__}")

check_versions()

# 全局配置
MODEL_DOWNLOAD_LIST = [
    "Illustrious_XL1.0.safetensors@https://www.modelscope.cn/models/ACCC1380/Illustrious-XL-v1.0.safetensors_20250212_2032/resolve/master/Illustrious-XL-v1.0.safetensors",
    "noob-eps1.1.safetensors@https://www.modelscope.cn/models/MusePublic/NoobAI-XL/resolve/epsilon1.1/file.safetensors",
    "noob-v1.0.safetensors@https://www.modelscope.cn/models/atonyxu/NoobAI-XL-Vpred/resolve/1.0/file.safetensors",
    "miaomiaoHarem_v15a.safetensors@https://www.modelscope.cn/models/ModelE/MiaoMiao-Harem/resolve/20250209130830/miaomiaoHarem_v15a.safetensors",
]
MODEL_DIR = "/root/ckpt"
EXPECTED_MODEL_SIZE_MB = 5000
MAX_RETRIES = 3

V_PREDICTION_MODELS = ["noob-v1.0.safetensors"]

SCHEDULERS = {
    "Euler a": EulerAncestralDiscreteScheduler,
    "Euler": EulerDiscreteScheduler,
    "DPM++ 2M": DPMSolverMultistepScheduler,
    "LMS": LMSDiscreteScheduler,
    "PNDM": PNDMScheduler,
    "DDIM": DDIMScheduler,
    "Heun": HeunDiscreteScheduler,
    "KDPM2": KDPM2DiscreteScheduler,
    "UniPC": UniPCMultistepScheduler,
    "DEIS": DEISMultistepScheduler
}

def ensure_executable(script_path):
    try:
        subprocess.run(["chmod", "+x", script_path], check=True)
        logger.info(f"Set executable permission for {script_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to set executable permission: {e.stderr.decode()}")

def check_file_size(file_path, expected_size_mb=EXPECTED_MODEL_SIZE_MB):
    if not os.path.exists(file_path):
        return False
    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if size_mb < expected_size_mb * 0.9:
        logger.warning(f"File {file_path} is incomplete, expected ~{expected_size_mb}MB, got {size_mb}MB")
        return False
    logger.info(f"File {file_path} size check passed: {size_mb}MB")
    return True

def download_models(download_list, model_dir, script_path="./download_with_aria2c.sh"):
    ensure_executable(script_path)
    os.makedirs(model_dir, exist_ok=True)
    for item in download_list:
        output_path = os.path.join(model_dir, os.path.basename(item.split("@")[0] if "@" in item else item))
        if os.path.exists(output_path) and check_file_size(output_path):
            logger.info(f"Model {os.path.basename(output_path)} already exists and is complete at {output_path}, skipping download.")
            continue
        
        for attempt in range(MAX_RETRIES):
            logger.info(f"Attempt {attempt + 1}/{MAX_RETRIES}: Downloading {item} to {model_dir}...")
            try:
                result = subprocess.run(
                    ["sudo", script_path, item, model_dir],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                logger.info(f"Download output: {result.stdout.decode()}")
                if not check_file_size(output_path):
                    raise ValueError(f"Downloaded file {output_path} is incomplete.")
                break
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to download {item}: {e.stderr.decode()}")
                if attempt == MAX_RETRIES - 1:
                    raise
                continue

download_models(MODEL_DOWNLOAD_LIST, MODEL_DIR)

def get_available_models(model_dir):
    models = [f for f in os.listdir(model_dir) if f.endswith(".safetensors")]
    if not models:
        raise ValueError(f"No models found in {model_dir}. Please ensure models are downloaded correctly.")
    return models

def load_model(model_path, scheduler_name="Euler a", use_v_prediction=False):
    global pipe, pipe_img2img
    try:
        pipe = StableDiffusionXLPipeline.from_single_file(
            model_path,
            use_safetensors=True,
            torch_dtype=torch.float16,
            variant="fp16"
        )
        pipe_img2img = StableDiffusionXLImg2ImgPipeline.from_single_file(
            model_path,
            use_safetensors=True,
            torch_dtype=torch.float16,
            variant="fp16"
        )
        
        scheduler_class = SCHEDULERS[scheduler_name]
        scheduler_config = pipe.scheduler.config
        scheduler_args = {"prediction_type": "v_prediction"} if use_v_prediction else {}
        pipe.scheduler = scheduler_class.from_config(scheduler_config, **scheduler_args)
        pipe_img2img.scheduler = scheduler_class.from_config(scheduler_config, **scheduler_args)
        
        try:
            pipe.enable_xformers_memory_efficient_attention()
            pipe_img2img.enable_xformers_memory_efficient_attention()
            logger.info("Enabled xformers for memory-efficient attention.")
        except Exception as e:
            logger.warning(f"xformers not available: {e}")
        
        pipe.enable_model_cpu_offload()
        pipe_img2img.enable_model_cpu_offload()
        pipe.enable_vae_slicing()
        pipe_img2img.enable_vae_slicing()
        
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
            pipe_img2img = pipe_img2img.to("cuda")
            logger.info(f"Models loaded to GPU with optimization from {model_path}.")
        else:
            pipe = pipe.to("cpu")
            pipe_img2img = pipe_img2img.to("cpu")
            logger.info(f"CUDA not available. Models loaded to CPU from {model_path}.")
        
        return pipe, pipe_img2img
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {str(e)}")
        raise

available_models = get_available_models(MODEL_DIR)
initial_model = os.path.join(MODEL_DIR, available_models[0])
pipe, pipe_img2img = load_model(initial_model)

def execute_command(command):
    try:
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return f"Output:\n{result.stdout.decode()}"
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr.decode()}"

def clear_system_memory():
    """清理系统内存"""
    try:
        gc.collect()  # 强制进行垃圾回收
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        logger.info(f"Current system memory usage: RSS={mem_info.rss / (1024 ** 2):.2f}MB, VMS={mem_info.vms / (1024 ** 2):.2f}MB")
        
        # 如果系统内存使用率过高，尝试释放
        if psutil.virtual_memory().percent > 90:
            logger.warning("High system memory usage detected, attempting to clear...")
            subprocess.run(["sync"], check=False)  # 同步文件系统缓冲区
            if sys.platform == "linux":
                subprocess.run(["sudo", "sysctl", "vm.drop_caches=3"], check=False)  # Linux 下释放缓存
            gc.collect()
            logger.info("System memory cleanup completed.")
    except Exception as e:
        logger.error(f"Failed to clear system memory: {str(e)}")

def check_and_recover_memory():
    """检查并恢复显存和系统内存"""
    try:
        memory_recovered = False
        
        # 检查并清理显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
            memory_reserved = torch.cuda.memory_reserved() / (1024 ** 3)    # GB
            logger.info(f"Current GPU memory: allocated={memory_allocated:.2f}GB, reserved={memory_reserved:.2f}GB")
            if memory_allocated > 0.9 * torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):
                logger.warning("High GPU memory usage detected, forcing cleanup...")
                torch.cuda.empty_cache()
                memory_recovered = True
        
        # 检查并清理系统内存
        clear_system_memory()
        if psutil.virtual_memory().percent > 90:
            logger.warning("High system memory usage detected during check.")
            memory_recovered = True
        
        return memory_recovered
    except Exception as e:
        logger.error(f"Memory check failed: {str(e)}")
        return True

def get_dimensions(active_tab, height_text, width_text, height_img, width_img):
    return (height_text, width_text) if active_tab == "文本到图像" else (height_img, width_img)

def generate_image(model_name, prompt, negative_prompt, height_text, width_text, height_img, width_img,
                  num_inference_steps, guidance_scale, seed, scheduler_name, use_v_prediction,
                  image_input_mode, init_image, init_image_base64, strength, active_tab):
    global pipe, pipe_img2img
    model_path = os.path.join(MODEL_DIR, model_name)
    
    default_positive = "1girl, best quality, very aesthetic, absurdres,"
    default_negative = "lowres, bad anatomy, bad hands, text, error, missing finger, extra digits, fewer digits, cropped, worst quality, low quality, low score, bad score, average score, signature, watermark, username, blurry"
    
    if not prompt.strip():
        prompt = default_positive
        logger.info("Prompt is empty, using default positive prompt.")
    if not negative_prompt.strip():
        negative_prompt = default_negative
        logger.info("Negative prompt is empty, using default negative prompt.")
    
    # 处理 CMD 指令
    cmd_output = handle_cmd(prompt)
    if cmd_output:
        return None, cmd_output
    
    current_v_pred = model_name in V_PREDICTION_MODELS
    use_v_prediction = use_v_prediction or current_v_pred
    if model_path != initial_model or pipe.scheduler.__class__ != SCHEDULERS[scheduler_name] or \
       current_v_pred != (pipe.scheduler.config.get("prediction_type") == "v_prediction"):
        logger.info(f"Switching to model {model_path} with {scheduler_name} scheduler...")
        pipe, pipe_img2img = load_model(model_path, scheduler_name, use_v_prediction)

    # 根据当前Tab选择宽高
    current_height, current_width = get_dimensions(active_tab, height_text, width_text, height_img, width_img)
    current_steps = num_inference_steps
    max_attempts = 3
    attempt = 0

    while attempt < max_attempts:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            clear_system_memory()  # 在生成前清理系统内存

            generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
            actual_seed = seed if seed != -1 else random.randint(0, 2**32 - 1)
            generator = generator.manual_seed(actual_seed)
            logger.info(f"Using seed: {actual_seed}")

            processed_image = None
            init_source = None
            
            if active_tab == "图像到图像":
                if image_input_mode == "图像上传" and init_image:
                    processed_image = init_image.convert("RGB")
                    init_source = "图像上传"
                elif image_input_mode == "Base64" and init_image_base64:
                    try:
                        if init_image_base64.startswith("data:image"):
                            init_image_base64 = init_image_base64.split(",")[1]
                        img_bytes = base64.b64decode(init_image_base64)
                        processed_image = Image.open(BytesIO(img_bytes)).convert("RGB")
                        init_source = "Base64"
                        logger.info("Converted Base64 image to PIL format.")
                    except Exception as e:
                        logger.error(f"Failed to decode Base64 image: {str(e)}")
                        return None, f"Error decoding Base64 image: {str(e)}"

            (
                prompt_embeds
                , prompt_neg_embeds
                , pooled_prompt_embeds
                , negative_pooled_prompt_embeds
            ) = get_weighted_text_embeddings_sdxl(
                pipe
                , prompt = prompt
                , neg_prompt = negative_prompt
            )

            if processed_image is not None:
                processed_image = processed_image.resize((current_width, current_height), Image.LANCZOS)
                with torch.no_grad():
                    image = pipe_img2img(
                        prompt_embeds = prompt_embeds,
                        negative_prompt_embeds = prompt_neg_embeds,
                        pooled_prompt_embeds = pooled_prompt_embeds,
                        negative_pooled_prompt_embeds = negative_pooled_prompt_embeds,
                        image=processed_image,
                        strength=strength,
                        num_inference_steps=current_steps,
                        guidance_scale=guidance_scale,
                        generator=generator,
                        height=current_height,
                        width=current_width
                    ).images[0]
                mode = "图像到图像"
            else:
                with torch.no_grad():
                    image = pipe(
                        prompt_embeds = prompt_embeds,
                        negative_prompt_embeds = prompt_neg_embeds,
                        pooled_prompt_embeds = pooled_prompt_embeds,
                        negative_pooled_prompt_embeds = negative_pooled_prompt_embeds,
                        height=current_height,
                        width=current_width,
                        num_inference_steps=current_steps,
                        guidance_scale=guidance_scale,
                        generator=generator
                    ).images[0]
                mode = "文本到图像"

            if check_and_recover_memory():
                logger.info("Memory cleanup performed after generation")

            output_message = (
                f"生成成功！\n"
                f"模式: {mode}\n"
                f"模型: {model_name}\n"
                f"提示词: {prompt}\n"
                f"反向提示词: {negative_prompt}\n"
                f"高度: {current_height}\n"
                f"宽度: {current_width}\n"
                f"步数: {current_steps}\n"
                f"引导系数: {guidance_scale}\n"
                f"种子: {actual_seed} (输入为 {seed})\n"
                f"调度器: {scheduler_name}\n"
                f"使用V预测: {use_v_prediction}\n"
                f"重绘幅度: {strength if processed_image is not None else 'N/A'}\n"
                f"初始图像来源: {init_source if init_source else '无'}"
            )
            
            return image, output_message

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            attempt += 1
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            clear_system_memory()  # 在出错后清理系统内存
            
            logger.warning(f"Memory error occurred (attempt {attempt}/{max_attempts}): {str(e)}")
            
            if current_height > MIN_HEIGHT:
                current_height = max(int(current_height * 0.8), MIN_HEIGHT)
            if current_width > MIN_WIDTH:
                current_width = max(int(current_width * 0.8), MIN_WIDTH)
            if current_steps > MIN_STEPS:
                current_steps = max(int(current_steps * 0.9), MIN_STEPS)
            
            logger.info(f"Adjusted parameters: height={current_height}, width={current_width}, steps={current_steps}")
            
            if attempt == max_attempts:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                clear_system_memory()  # 最后一次失败后清理系统内存
                return None, f"Failed to generate image after {max_attempts} attempts: {str(e)}\n" \
                            f"Final parameters: height={current_height}, width={current_width}, steps={current_steps}\n" \
                            f"Memory has been cleaned up for the next attempt."

def update_image_input_visibility(mode):
    if mode == "图像上传":
        return gr.update(visible=True), gr.update(visible=False)
    elif mode == "Base64":
        return gr.update(visible=False), gr.update(visible=True)
    else:
        return gr.update(visible=False), gr.update(visible=False)

html_content = """
<div style="position: fixed; bottom: 0; left: 0; right: 0; background-color: rgba(248, 248, 255, 0.9); text-align: center; padding: 10px; border-top: 1px solid #e6e6ff;">
    <span style="font-size: 14px; color: #6c5ce7;">Author: spawner | </span>
    <a href='https://github.com/spawner1145' target='_blank' style='font-size: 14px; color: #8c7ae6; text-decoration: none;'>GitHub</a>
    <style>
        #copyright a:hover {text-decoration: underline; color: #7f6aae;}
    </style>
</div>
"""

# 处理 CMD 指令的独立函数，确保立即执行
def handle_cmd_input(prompt):
    if prompt.strip().startswith("cmd "):
        command = prompt.replace("cmd ", "", 1)
        logger.info(f"Executing command immediately: {command}")
        return None, execute_command(command)
    return None, None  # 不处理非 CMD 输入

with gr.Blocks(title="sdxl系图像生成器", 
               css="#generate_btn {margin-top: 10px;}") as demo:
    gr.Markdown("# sdxl系图像生成器")
    default_positive = "1girl, best quality, very aesthetic, absurdres,"
    default_negative = "lowres, bad anatomy, bad hands, text, error, missing finger, extra digits, fewer digits, cropped, worst quality, low quality, low score, bad score, average score, signature, watermark, username, blurry"
    
    active_tab_state = gr.State(value="文本到图像")
    
    with gr.Row():
        with gr.Column():
            model_dropdown = gr.Dropdown(label="选择模型", choices=available_models, value=available_models[0])
            prompt = gr.Textbox(label="提示词", placeholder="正面提示词", lines=2, value=default_positive)
            negative_prompt = gr.Textbox(label="负面提示词", placeholder="负面提示词（可选）", lines=2, value=default_negative)
            
            with gr.Tab("文本到图像") as text_tab:
                height_text = gr.Slider(256, 2048, value=DEFAULT_HEIGHT, step=64, label="高度")
                width_text = gr.Slider(256, 2048, value=DEFAULT_WIDTH, step=64, label="宽度")
            
            with gr.Tab("图像到图像") as img_tab:
                image_input_mode = gr.Dropdown(
                    label="图像传入模式",
                    choices=["图像上传", "Base64"],
                    value="图像上传"
                )
                init_image = gr.Image(label="初始图像（图像上传）", type="pil", visible=True)
                init_image_base64 = gr.Textbox(label="初始图像（Base64）", placeholder="粘贴Base64编码的图像数据", lines=3, visible=False)
                height_img = gr.Slider(256, 2048, value=DEFAULT_HEIGHT, step=64, label="输出高度")
                width_img = gr.Slider(256, 2048, value=DEFAULT_WIDTH, step=64, label="输出宽度")
                strength = gr.Slider(0.1, 1.0, value=0.8, step=0.1, label="重绘幅度")
            
            steps = gr.Slider(10, 100, value=DEFAULT_STEPS, step=1, label="步数(steps)")
            guidance = gr.Slider(1, 20, value=7.5, step=0.5, label="提示词引导系数(cfg)")
            scheduler_dropdown = gr.Dropdown(
                label="采样方法",
                choices=list(SCHEDULERS.keys()),
                value="Euler a"
            )
            use_v_prediction = gr.Checkbox(label="使用V预测", value=False)
            seed = gr.Number(value=-1, label="随机种子（-1 为随机）", precision=0)
            generate_btn = gr.Button("生成图像", elem_id="generate_btn")
        
        with gr.Column():
            output_image = gr.Image(label="生成的图像", type="pil")
            output_text = gr.Textbox(label="状态信息", lines=10)

    def update_v_prediction(model_name):
        return model_name in V_PREDICTION_MODELS
    
    def update_tab_state(tab_name):
        return tab_name

    model_dropdown.change(
        fn=update_v_prediction,
        inputs=[model_dropdown],
        outputs=[use_v_prediction]
    )

    image_input_mode.change(
        fn=update_image_input_visibility,
        inputs=[image_input_mode],
        outputs=[init_image, init_image_base64]
    )

    text_tab.select(fn=lambda: "文本到图像", inputs=None, outputs=active_tab_state)
    img_tab.select(fn=lambda: "图像到图像", inputs=None, outputs=active_tab_state)

    def handle_cmd(prompt):
        if prompt.strip().startswith("cmd "):
            command = prompt.replace("cmd ", "", 1)
            logger.info(f"Executing command: {command}")
            return execute_command(command)
        return None

    # 生成按钮点击事件
    generate_btn.click(
        fn=generate_image,
        inputs=[model_dropdown, prompt, negative_prompt, 
                height_text, width_text, height_img, width_img,
                steps, guidance, seed, scheduler_dropdown, use_v_prediction,
                image_input_mode, init_image, init_image_base64, strength, active_tab_state],
        outputs=[output_image, output_text]
    )

    # 提示词提交事件
    prompt.submit(
        fn=generate_image,
        inputs=[model_dropdown, prompt, negative_prompt,
                height_text, width_text, height_img, width_img,
                steps, guidance, seed, scheduler_dropdown, use_v_prediction,
                image_input_mode, init_image, init_image_base64, strength, active_tab_state],
        outputs=[output_image, output_text]
    )
    gr.HTML(html_content)

logger.info("Starting Gradio application...")
demo.launch(server_name="0.0.0.0", server_port=7860)
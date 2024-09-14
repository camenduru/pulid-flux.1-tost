import os, json, requests, runpod

import time
import torch
from einops import rearrange
from PIL import Image
import numpy as np

from flux.cli import SamplingOptions
from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack
from flux.util import load_ae, load_clip, load_flow_model, load_t5
from pulid.pipeline_flux import PuLIDPipeline
from pulid.utils import resize_numpy_image_long

def get_models(name: str, device: torch.device, offload: bool):
    t5 = load_t5(device, max_length=128)
    clip = load_clip(device)
    model = load_flow_model(name, device="cpu" if offload else device)
    model.eval()
    ae = load_ae(name, device="cpu" if offload else device)
    return model, ae, t5, clip

class FluxGenerator:
    def __init__(self):
        self.device = torch.device('cuda')
        self.offload = False
        self.model_name = 'flux-dev'
        self.model, self.ae, self.t5, self.clip = get_models(
            self.model_name,
            device=self.device,
            offload=self.offload,
        )
        self.pulid_model = PuLIDPipeline(self.model, 'cuda', weight_dtype=torch.bfloat16)
        self.pulid_model.load_pretrain()

flux_generator = FluxGenerator()

@torch.inference_mode()
def generate_image(
        width,
        height,
        num_steps,
        start_step,
        guidance,
        seed,
        prompt,
        id_image=None,
        id_weight=1.0,
        neg_prompt="",
        true_cfg=1.0,
        timestep_to_start_cfg=1,
        max_sequence_length=128,
):
    flux_generator.t5.max_length = max_sequence_length

    seed = int(seed)
    if seed == -1:
        seed = None

    opts = SamplingOptions(
        prompt=prompt,
        width=width,
        height=height,
        num_steps=num_steps,
        guidance=guidance,
        seed=seed,
    )

    if opts.seed is None:
        opts.seed = torch.Generator(device="cpu").seed()
    print(f"Generating '{opts.prompt}' with seed {opts.seed}")
    t0 = time.perf_counter()

    use_true_cfg = abs(true_cfg - 1.0) > 1e-2

    if id_image is not None:
        id_image = resize_numpy_image_long(id_image, 1024)
        id_embeddings, uncond_id_embeddings = flux_generator.pulid_model.get_id_embedding(id_image, cal_uncond=use_true_cfg)
    else:
        id_embeddings = None
        uncond_id_embeddings = None

    print(id_embeddings)

    # prepare input
    x = get_noise(
        1,
        opts.height,
        opts.width,
        device=flux_generator.device,
        dtype=torch.bfloat16,
        seed=opts.seed,
    )
    print(x)
    timesteps = get_schedule(
        opts.num_steps,
        x.shape[-1] * x.shape[-2] // 4,
        shift=True,
    )

    if flux_generator.offload:
        flux_generator.t5, flux_generator.clip = flux_generator.t5.to(flux_generator.device), flux_generator.clip.to(flux_generator.device)
    inp = prepare(t5=flux_generator.t5, clip=flux_generator.clip, img=x, prompt=opts.prompt)
    inp_neg = prepare(t5=flux_generator.t5, clip=flux_generator.clip, img=x, prompt=neg_prompt) if use_true_cfg else None

    # offload TEs to CPU, load model to gpu
    if flux_generator.offload:
        flux_generator.t5, flux_generator.clip = flux_generator.t5.cpu(), flux_generator.clip.cpu()
        torch.cuda.empty_cache()
        flux_generator.model = flux_generator.model.to(flux_generator.device)

    # denoise initial noise
    x = denoise(
        flux_generator.model, **inp, timesteps=timesteps, guidance=opts.guidance, id=id_embeddings, id_weight=id_weight,
        start_step=start_step, uncond_id=uncond_id_embeddings, true_cfg=true_cfg,
        timestep_to_start_cfg=timestep_to_start_cfg,
        neg_txt=inp_neg["txt"] if use_true_cfg else None,
        neg_txt_ids=inp_neg["txt_ids"] if use_true_cfg else None,
        neg_vec=inp_neg["vec"] if use_true_cfg else None,
    )

    # offload model, load autoencoder to gpu
    if flux_generator.offload:
        flux_generator.model.cpu()
        torch.cuda.empty_cache()
        flux_generator.ae.decoder.to(x.device)

    # decode latents to pixel space
    x = unpack(x.float(), opts.height, opts.width)
    with torch.autocast(device_type=flux_generator.device.type, dtype=torch.bfloat16):
        x = flux_generator.ae.decode(x)

    if flux_generator.offload:
        flux_generator.ae.decoder.cpu()
        torch.cuda.empty_cache()

    t1 = time.perf_counter()

    print(f"Done in {t1 - t0:.1f}s.")
    # bring into PIL format
    x = x.clamp(-1, 1)
    # x = embed_watermark(x.float())
    x = rearrange(x[0], "c h w -> h w c")

    img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
    return img, str(opts.seed), flux_generator.pulid_model.debug_img_list

def closestNumber(n, m):
    q = int(n / m)
    n1 = m * q
    if (n * m) > 0:
        n2 = m * (q + 1)
    else:
        n2 = m * (q - 1)
    if abs(n - n1) < abs(n - n2):
        return n1
    return n2

def download_file(url, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    file_name = url.split('/')[-1]
    file_path = os.path.join(save_dir, file_name)
    response = requests.get(url)
    response.raise_for_status()
    with open(file_path, 'wb') as file:
        file.write(response.content)
    return file_path

@torch.inference_mode()
def generate(input):
    values = input["input"]

    final_width = values['final_width']
    id_image = values['input_image_check']
    id_image = download_file(url=id_image, save_dir='/content')
    id_image = Image.open(id_image)
    width, height = id_image.size
    id_image = np.array(id_image)
    image_aspect_ratio = width / height
    height = final_width / image_aspect_ratio
    num_steps = values['num_steps']
    start_step = values['start_step']
    guidance = values['guidance']
    seed = values['seed']
    prompt = values['prompt']
    neg_prompt = values['neg_prompt']
    id_weight = values['id_weight']
    true_cfg = values['true_cfg']
    timestep_to_start_cfg = values['timestep_to_start_cfg']
    max_sequence_length = values['max_sequence_length']

    if seed == 0:
        random.seed(int(time.time()))
        seed = random.randint(0, 18446744073709551615)
    print(seed)

    generate_image(
        width=final_width,
        height=height,
        num_steps=num_steps,
        start_step=start_step,
        guidance=guidance,
        seed=seed,
        prompt=prompt,
        id_image=id_image,
        id_weight=id_weight,
        neg_prompt=neg_prompt,
        true_cfg=true_cfg,
        timestep_to_start_cfg=timestep_to_start_cfg,
        max_sequence_length=max_sequence_length,
    )[0].save("/content/pulid-flux-tost.png")

    result = "/content/pulid-flux-tost.png"
    try:
        notify_uri = values['notify_uri']
        del values['notify_uri']
        notify_token = values['notify_token']
        del values['notify_token']
        discord_id = values['discord_id']
        del values['discord_id']
        if(discord_id == "discord_id"):
            discord_id = os.getenv('com_camenduru_discord_id')
        discord_channel = values['discord_channel']
        del values['discord_channel']
        if(discord_channel == "discord_channel"):
            discord_channel = os.getenv('com_camenduru_discord_channel')
        discord_token = values['discord_token']
        del values['discord_token']
        if(discord_token == "discord_token"):
            discord_token = os.getenv('com_camenduru_discord_token')
        job_id = values['job_id']
        del values['job_id']
        default_filename = os.path.basename(result)
        with open(result, "rb") as file:
            files = {default_filename: file.read()}
        payload = {"content": f"{json.dumps(values)} <@{discord_id}>"}
        response = requests.post(
            f"https://discord.com/api/v9/channels/{discord_channel}/messages",
            data=payload,
            headers={"Authorization": f"Bot {discord_token}"},
            files=files
        )
        response.raise_for_status()
        result_url = response.json()['attachments'][0]['url']
        notify_payload = {"jobId": job_id, "result": result_url, "status": "DONE"}
        web_notify_uri = os.getenv('com_camenduru_web_notify_uri')
        web_notify_token = os.getenv('com_camenduru_web_notify_token')
        if(notify_uri == "notify_uri"):
            requests.post(web_notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
        else:
            requests.post(web_notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
            requests.post(notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": notify_token})
        return {"jobId": job_id, "result": result_url, "status": "DONE"}
    except Exception as e:
        error_payload = {"jobId": job_id, "status": "FAILED"}
        try:
            if(notify_uri == "notify_uri"):
                requests.post(web_notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
            else:
                requests.post(web_notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
                requests.post(notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": notify_token})
        except:
            pass
        return {"jobId": job_id, "result": f"FAILED: {str(e)}", "status": "FAILED"}
    finally:
        if os.path.exists(result):
            os.remove(result)

runpod.serverless.start({"handler": generate})
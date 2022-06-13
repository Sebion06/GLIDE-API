import sys
import requests
import argparse


def get_request_json_data(url, type):
    if type == "GET":
        data = requests.get(url)
    elif type == "POST":
        data = requests.post(url)
    return data.json()


def get_request(args):
    request = f"{args.ip}:{args.port}/image?prompt={args.prompt}&size={args.size}"
    if args.batch_size is not None:
        request += f"&batch_size={args.batch_size}"
    if args.guidance_scale is not None:
        request += f"&guidance_scale={args.guidance_scale}"
    if args.upsample_temp is not None:
        request += f"&upsample_temp={args.upsample_temp}"
    
    data = get_request_json_data(request, "GET")
    return data


def post_request(args):
    request = f"{args.ip}:{args.port}/"
    if args.base_diffusion_steps is not None:
        new_request = f"{request}basemodel?diffusion_steps={args.base_diffusion_steps}"
        data = get_request_json_data(new_request, "POST")
        print(f"Updated base model with {args.base_diffusion_steps} diffusion steps and {data['base_model_total_parameters']} total parameters")
    if args.up_diffusion_steps is not None:
        new_request = f"{request}upmodel?diffusion_steps={args.up_diffusion_steps}"
        data = get_request_json_data(new_request, "POST")
        print(f"Updated upsample model with {args.up_diffusion_steps} diffusion steps and {data['upsample_model_total_parameters']} total parameters")


def save_html_file(prompt, data):
    html_content = f"""
    <html>
        <body>
            <h2>Prompt: {prompt}; Size: {data["image_size"]}</h2>
            <img src="data:{data["data_type"]},{data["image"]}" alt="{prompt}" />
        </body>
    </html>
    """
    with open('test.html', 'w+') as f:
        f.write(html_content)


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--ip",  action="store", help="IP of the server on which the python app is hosted", required=True)
    parser.add_argument("--port",  action="store", help="Port through which the server can be accesed", required=True)
    parser.add_argument("--prompt",  action="store", help="Text prompt for the generating image", default=None)
    parser.add_argument("-r", "--request", choices=['GET', 'POST'], help="Request type", required=True)
    parser.add_argument("-s", "--size", action="store", help="Size of the picture requested", default=None)
    parser.add_argument("-b", "--batch_size", help="Batch size requested. Used with GET req. Recommended: 1", default=None)
    parser.add_argument("-g", "--guidance_scale", help="Guidance scale of the model. Used with GET req. Recommended: 3.0", default=None)
    parser.add_argument("-u", "--upsample_temp", help="Upsample temp of the model. Used with GET req. Values between 0 and 1. Recommended: 0.997", default=None)
    parser.add_argument("-bds", "--base_diffusion_steps", help="Diffusion steps for base model. Used with POST req", default=None)
    parser.add_argument("-uds", "--up_diffusion_steps", help="Diffusion steps for up model. Used with POST req", default=None)
    args = parser.parse_args()
    if args.request == "GET" and args.size is None and args.prompt is None:
        print("size and prompt args must be supplied with GET request")
        sys.exit(1)
    if args.request == "POST" and args.base_diffusion_steps is None and args.up_diffusion_steps is None:
        print("base_diffusion_steps or up_diffusion_steps arg must be supplied with POST request")
        sys.exit(1)
    return args


def main():
    args = parse_arguments()
    if args.request == "GET":
        data = get_request(args)
        save_html_file(args.prompt, data)
    elif args.request == "POST":
        post_request(args)

if __name__ == "__main__":
    main()
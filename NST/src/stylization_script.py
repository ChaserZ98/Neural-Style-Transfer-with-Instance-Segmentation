import argparse
import os

import cv2 as cv
import torch
import NST.utils.utils as utils
from NST.models.definitions.transformer_net import TransformerNet
from torch.utils.data import DataLoader


def stylize_static_image(inference_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare the model - load the weights and put the model into evaluation mode
    stylization_model = TransformerNet().to(device)
    training_state = torch.load(os.path.join(inference_config["model_binaries_path"], inference_config["model_name"]))
    state_dict = training_state["state_dict"]
    stylization_model.load_state_dict(state_dict, strict=True)
    stylization_model.eval()

    if inference_config['verbose']:
        utils.print_model_metadata(training_state)

    with torch.no_grad():
        if os.path.isdir(os.path.join(inference_config['content_images_path'], inference_config['content_input'])):  # do a batch stylization (every image in the directory)
            for img_name in os.listdir(os.path.join(inference_config['content_images_path'], inference_config['content_input'])):
                content_img_path = os.path.join(inference_config['content_images_path'], inference_config['content_input'], img_name)
                content_image = utils.prepare_img(content_img_path, inference_config['img_width'], device)
                target_shape = cv.imread(content_img_path).shape[::-1][1:]
                stylized_img = stylization_model(content_image).to('cpu').numpy()[0]
                utils.save_and_maybe_display_image(inference_config, stylized_img, target_shape, should_display=inference_config['should_not_display'])

            # utils.save_and_maybe_display_image(inference_config, stylized_img, target_shape, should_display=inference_config['should_not_display'])
            
            # img_dataset = utils.SimpleDataset(os.path.join(content_images_path, inference_config['content_input']), inference_config['img_width'])
            # img_loader = DataLoader(img_dataset, batch_size=1)

            # try:
            #     processed_imgs_cnt = 0
            #     for batch_id, img_batch in enumerate(img_loader):
            #         processed_imgs_cnt += len(img_batch)
            #         if inference_config['verbose']:
            #             print(f'Processing batch {batch_id + 1} ({processed_imgs_cnt}/{len(img_dataset)} processed images).')

            #         img_batch = img_batch.to(device)
            #         stylized_imgs = stylization_model(img_batch).to('cpu').numpy()
            #         # print(img_batch.shape)
            #         for img_id, stylized_img in enumerate(stylized_imgs):
            #             # print(stylized_img.shape)
            #             # stylized_img = utils.Resize(600)(stylized_img)
            #             # print(stylized_img.shape)
            #             # print(img_batch[img_id][0].shape[::-1])
            #             utils.save_and_maybe_display_image(inference_config, stylized_img, target_shape=img_batch[img_id][0].shape[::-1], should_display=False)
            # except Exception as e:
            #     print(e)
            #     print(f'Consider making the batch_size (current = {inference_config["batch_size"]} images) or img_width (current = {inference_config["img_width"]} px) smaller')
            #     exit(1)

        else:  # do stylization for a single image
            content_img_path = os.path.join(inference_config['content_images_path'], inference_config['content_input'])
            content_image = utils.prepare_img(content_img_path, inference_config['img_width'], device)
            target_shape = content_image[0][0].shape[::-1]
            stylized_img = stylization_model(content_image).to('cpu').numpy()[0]

            utils.save_and_maybe_display_image(inference_config, stylized_img, target_shape, should_display=inference_config['should_not_display'])


if __name__ == "__main__":
    #
    # Fixed args - don't change these unless you have a good reason
    #
    content_images_path = os.path.join(os.path.dirname(__file__), 'data', 'content-images')
    output_images_path = os.path.join(os.path.dirname(__file__), 'data', 'output-images')
    model_binaries_path = os.path.join(os.path.dirname(__file__), 'models', 'binaries')

    assert utils.dir_contains_only_models(model_binaries_path), f'Model directory should contain only model binaries.'
    os.makedirs(output_images_path, exist_ok=True)

    #
    # Modifiable args - feel free to play with these
    #
    parser = argparse.ArgumentParser()
    # Put image name or directory containing images (if you'd like to do a batch stylization on all those images)
    parser.add_argument("--content_input", type=str, help="Content image(s) to stylize", default='000072.jpg')
    parser.add_argument("--batch_size", type=int, help="Batch size used only if you set content_input to a directory", default=5)
    parser.add_argument("--img_width", type=int, help="Resize content image to this width", default=None)
    parser.add_argument("--model_name", type=str, help="Model binary to use for stylization", default='style_madhubani_datapoints_508100_cw_1.0_sw_400000.0_tw_0.pth')

    # Less frequently used arguments
    parser.add_argument("--should_not_display", action='store_false', help="Should display the stylized result", default=False)
    parser.add_argument("--verbose", action='store_true', help="Print model metadata (how the model was trained) and where the resulting stylized image was saved")
    parser.add_argument("--redirected_output", type=str, help="Overwrite default output dir. Useful when this project is used as a submodule", default="data/output-images/NST_example_output/vg_starry_night")
    parser.add_argument("--use_origin_name", type=str, help="Use the origin file name as the output img file name", default=True)
    args = parser.parse_args()

    # if redirected output is not set when doing batch stylization set to default image output location
    if os.path.isdir(os.path.join(content_images_path, args.content_input)) and args.redirected_output is None:
        args.redirected_output = output_images_path

    # Wrapping inference configuration into a dictionary
    inference_config = dict()
    for arg in vars(args):
        inference_config[arg] = getattr(args, arg)
    inference_config['content_images_path'] = content_images_path
    inference_config['output_images_path'] = output_images_path
    inference_config['model_binaries_path'] = model_binaries_path

    stylize_static_image(inference_config)


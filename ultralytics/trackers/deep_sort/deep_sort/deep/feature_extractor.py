import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import logging

from .model import Net


class Extractor(object):
    def __init__(self, model_path, use_cuda=True):
        self.net = Net(reid=True)
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        state_dict = torch.load(model_path, map_location=torch.device(self.device))['net_dict']
        self.net.load_state_dict(state_dict)
        logger = logging.getLogger("root.tracker")
        logger.info("Loading weights from {}... Done!".format(model_path))
        self.net.to(self.device)
        self.size = (64, 128)
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _preprocess(self, im_crops):
        
        def _resize(im, size):
            # Convert to float32 and normalize
            im_float = im.astype(np.float32) / 255.0
            
            try:
                resized = cv2.resize(im_float, size)
                return resized
            except cv2.error as e:
                raise ValueError(f"Failed to resize image: {str(e)}")

        try:
            # Process each image crop with validation
            processed_crops = []
            for im in im_crops:
                try:
                    resized = _resize(im, self.size)
                    normalized = self.norm(resized)
                    processed_crops.append(normalized.unsqueeze(0))
                except (ValueError, cv2.error) as e:
                    logging.warning(f"Failed to process image crop: {str(e)}")
                    continue
            
            if not processed_crops:
                raise ValueError("No valid image crops to process")
                
            # Concatenate all processed crops
            im_batch = torch.cat(processed_crops, dim=0).float()
            return im_batch
            
        except Exception as e:
            logging.error(f"Error in preprocessing: {str(e)}")
            raise

    def __call__(self, im_crops):
        try:
            im_batch = self._preprocess(im_crops)
            with torch.no_grad():
                im_batch = im_batch.to(self.device)
                features = self.net(im_batch)
            return features.cpu().numpy()
        except Exception as e:
            logging.error(f"Error in feature extraction: {str(e)}")
            raise

if __name__ == '__main__':
    img = cv2.imread("demo.jpg")[:, :, (2, 1, 0)]
    extr = Extractor("checkpoint/ckpt.t7")
    feature = extr(img)
    print(feature.shape)
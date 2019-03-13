import click
import cv2
import onegan
import torch
import torchvision.transforms as T
from PIL import Image
from pathlib import Path
from matplotlib import pyplot as plt
import os

#from trainer.賣扣老師 import build_resnet101_FCN
from trainer.model import ResPlanarSeg

torch.backends.cudnn.benchmark = True


class Predictor:

    def __init__(self, input_size, weight=None):
        self.model = self.build_model(weight)
        self.colorizer = onegan.extension.Colorizer(
            colors=[
                [249, 69, 93], [255, 229, 170], [144, 206, 181],
                [81, 81, 119], [241, 247, 210]])
        self.transform = T.Compose([
            T.Resize(input_size),
            T.ToTensor(),
            T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
        ])

    def build_model(self, weight_path, joint_class=False):
        #model = build_resnet101_FCN(pretrained=False, nb_classes=37, stage_2=True, joint_class=joint_class)
        model = ResPlanarSeg(num_classes=5, pretrained=True)
        weight = onegan.utils.export_checkpoint_weight(weight_path)
        model.load_state_dict(weight)
        model.eval()
        return model.cuda()

    @onegan.utils.timeit
    def process(self, raw, f):

        def _batched_process(batched_img):
            score, pred_type = self.model(onegan.utils.to_var(batched_img))
            pred_type = torch.argmax(pred_type, 1)
            '''
            if pred_type.item() == 0:
                print('-------------(%s: %d)--------------' % (f, pred_type.item()))
            '''
            _, output = torch.max(score, 1)
            image = (batched_img / 2 + .5)
            layout = self.colorizer.apply(output.data.cpu())
            #return image * .6 + layout * .4
            return layout, pred_type.item()

        img = Image.fromarray(raw)
        batched_img = self.transform(img).unsqueeze(0)
        canvas, t = _batched_process(batched_img)
        result = canvas.squeeze().permute(1, 2, 0).numpy()
        return cv2.resize(result, (raw.shape[1], raw.shape[0])), t


@click.command()
@click.option('--input_path', type=click.Path(exists=True), default='../data/test_input')
@click.option('--output_path', type=click.Path(exists=True), default='../data/test_output')
@click.option('--weight', type=click.Path(exists=True))
@click.option('--input_size', default=(320, 320), type=(int, int))
def main(input_path, output_path, weight, input_size):

    demo = Predictor(input_size, weight=weight)

    input_path = Path(input_path)
    output_path = Path(output_path)
    with open('./pred_type_0.txt', 'a') as type_f:
        for f in os.listdir(input_path):
            try:
                output, t = demo.process(cv2.imread(str(input_path / f))[:, :, ::-1], f)
                if t == 0:
                    plt.imsave(output_path / f, output)
                    type_f.write('%s\n' % f)
            except:
                print('exp: ', f)

if __name__ == '__main__':
    main()

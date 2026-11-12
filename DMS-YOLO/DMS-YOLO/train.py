
import argparse
from ultralytics import YOLO

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, help='dataset YAML, YOLO detection format')
    p.add_argument('--epochs', type=int, default=150)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--device', default=None)
    p.add_argument('--weights', default=None, help='optional pretrained *.pt with compatible architecture')
    a = p.parse_args()
    model = YOLO('ultralytics/cfg/models/dms-yolo11.yaml')
    if a.weights:
        model.load(a.weights)
    args = dict(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, project='runs/dms-yolo', name='train')
    if a.device is not None:
        args['device'] = a.device
    model.train(**args)

if __name__ == '__main__':
    main()

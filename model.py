from PIL import Image
from torchvision import transforms


def preprocess_numpy(img: Image) -> Image:
    # import preprocessing function from pytorch_model.Classifier class
    resize = transforms.Resize((224, 224))
    crop = transforms.CenterCrop((224, 224))
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    img = resize(img)
    img = crop(img)
    img = to_tensor(img)
    img = normalize(img)
    return img


if __name__ == "__main__":
    img = Image.open("./resources/n01667114_mud_turtle.jpeg")
    img = preprocess_numpy(img).unsqueeze(0)
    print(img)

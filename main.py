# standard library
import shutil
from typing import List

# third-party
import cv2
from dotenv import load_dotenv
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image


def load_model(lang: str = "ch") -> PaddleOCR:
    """
    PaddleOCR的参数通过在安装了paddleocr库的虚拟环境中，通过命令`paddleocr --help`进行查看
    det_model_dir = ""
    rec_model_dir = ""
    cls_model_dir = ""
    sr_model_dir = ""
    ser_model_dir = ""
    re_model_dir = ""
    """
    # load OCR model
    ocr_model = PaddleOCR(
        lang=lang
    )  # need to run only once to download and load model into memory
    return ocr_model


def preprocess(img_path: str):
    shutil.copy(img_path, f"{img_path}.bak")  # make a backup
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 灰度化
    # _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)  # 二值化
    cv2.imwrite(img_path, img)


def run_ocr(ocr_model: PaddleOCR, img_path: str) -> List[List[str]]:
    preprocess(img_path)
    result = ocr_model.ocr(img_path, cls=False)
    return result


def show_results(result: List[List[str]]) -> None:
    print(f"{'=' * 40} [Print Result] {'=' * 40}")
    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            print(line)


def draw_result(
    img_path: str,
    result: List[str],
    save_filename: str = "./logs/result.png",
    font_path: str = "./fonts/simfang.ttf",
) -> None:
    # draw result
    image = Image.open(img_path).convert("RGB")
    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    im_show = draw_ocr(image, boxes, txts, scores, font_path=font_path)
    im_show = Image.fromarray(im_show)
    im_show.save(save_filename)


def main():
    # load environment variables
    load_dotenv("ocr.env")
    # prepare
    ocr_model = load_model(lang="ch")
    img_path = "./figs/yyslz_djq.png"
    # run OCR!
    result = run_ocr(ocr_model, img_path)
    # print results
    show_results(result)
    # draw results
    draw_result(
        img_path,
        result[0],
        save_filename="logs/result.png",
        font_path="./fonts/simfang.ttf",
    )


if __name__ == "__main__":
    main()

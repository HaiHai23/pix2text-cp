# Various models
**Pix2Text (P2T)** integrates many different functional models, mainly including:

- **Layout Analysis Model**: [breezedeus/pix2text-layout](https://huggingface.co/breezedeus/pix2text-layout) ([Domestic Mirror](https://hf-mirror.com/breezedeus/pix2text-layout)).
- **Table Recognition Model**: [breezedeus/pix2text-table-rec](https://huggingface.co/breezedeus/pix2text-table-rec) ([Domestic Mirror](https://hf-mirror.com/breezedeus/pix2text-table-rec)).
- **Text Recognition Engine**: Supports **`80+` languages**, such as **English, Simplified Chinese, Traditional Chinese, Vietnamese**, etc. Among them, **English** and **Simplified Chinese** recognition uses the open-source OCR tool [CnOCR](https://github.com/breezedeus/cnocr), while other languages use the open-source OCR tool [EasyOCR](https://github.com/JaidedAI/EasyOCR).
- **Mathematical Formula Detection Model (MFD)**: [breezedeus/pix2text-mfd](https://huggingface.co/breezedeus/pix2text-mfd) ([Domestic Mirror](https://hf-mirror.com/breezedeus/pix2text-mfd)). Implemented based on [CnSTD](https://github.com/breezedeus/cnstd).
- **Mathematical Formula Recognition Model (MFR)**: [breezedeus/pix2text-mfr](https://huggingface.co/breezedeus/pix2text-mfr) ([Domestic Mirror](https://hf-mirror.com/breezedeus/pix2text-mfr)).

Many of these models come from other open-source authors, and we are very grateful for their contributions.

These models will normally be downloaded automatically (it may be slow, please do not interrupt the download process if there are no errors), but if the download fails, you can refer to the instructions below to download them manually.

In addition to the basic models, Pix2Text also offers advanced paid versions of the following models:

- Paid MFD and MFR models: For details, refer to [P2T Detailed Information | Breezedeus.com](https://www.breezedeus.com/article/pix2text_cn).
- Paid CnOCR models: For details, refer to [CnOCR Detailed Information | Breezedeus.com](https://www.breezedeus.com/article/cnocr).

For specific instructions, please see the end of this page.

The following instructions are mainly for the free basic models.

## Layout Analysis Model
**Layout Analysis Model** download address: [breezedeus/pix2text-layout](https://huggingface.co/breezedeus/pix2text-layout) (use [Domestic Mirror](https://hf-mirror.com/breezedeus/pix2text-layout) if you cannot access the main site).
Download all files from here to `~/.pix2text/1.1/layout-parser` (for Windows, place them in `C:\Users\<username>\AppData\Roaming\pix2text\1.1\layout-parser`). Create the directory if it does not exist.

> Note: The `1.1` in the path above is the version number of pix2text, `1.1.*` corresponds to `1.1`. If it is another version, please replace it accordingly.

## Table Recognition Model
**Table Recognition Model** download address: [breezedeus/pix2text-table-rec](https://huggingface.co/breezedeus/pix2text-table-rec) (use [Domestic Mirror](https://hf-mirror.com/breezedeus/pix2text-table-rec) if you cannot access the main site).
Download all files from here to `~/.pix2text/1.1/table-rec` (for Windows, place them in `C:\Users\<username>\AppData\Roaming\pix2text\1.1\table-rec`). Create the directory if it does not exist.

> Note: The `1.1` in the path above is the version number of pix2text, `1.1.*` corresponds to `1.1`. If it is another version, please replace it accordingly.

## Mathematical Formula Detection Model (MFD)
### `pix2text >= 1.1.1`
Starting from **V1.1.1**, the **Mathematical Formula Detection Model** download address is: [breezedeus/pix2text-mfd](https://huggingface.co/breezedeus/pix2text-mfd) (use [Domestic Mirror](https://hf-mirror.com/breezedeus/pix2text-mfd) if you cannot access the main site).

### `pix2text < 1.1.1`
The **Mathematical Formula Detection Model** (MFD) comes from the [CnSTD](https://github.com/breezedeus/cnstd) mathematical formula detection model (MFD). Please refer to its repository for instructions.

If the system cannot automatically download the model files successfully, you need to manually download them from the [**cnstd-cnocr-models**](https://huggingface.co/breezedeus/cnstd-cnocr-models) ([Domestic Mirror](https://hf-mirror.com/breezedeus/cnstd-cnocr-models)) project, or from [Baidu Cloud](https://pan.baidu.com/s/1zDMzArCDrrXHWL0AWxwYQQ?pwd=nstd) (extraction code: `nstd`). Place the downloaded zip file in the `~/.cnstd/1.2` directory (for Windows, `C:\Users\<username>\AppData\Roaming\cnstd\1.2`).

## Mathematical Formula Recognition Model (MFR)
**Mathematical Formula Recognition Model** download address: [breezedeus/pix2text-mfr](https://huggingface.co/breezedeus/pix2text-mfr) (use [Domestic Mirror](https://hf-mirror.com/breezedeus/pix2text-mfr) if you cannot access the main site).
Download all files from here to `~/.pix2text/1.1/mfr-onnx` (for Windows, place them in `C:\Users\<username>\AppData\Roaming\pix2text\1.1\mfr-onnx`). Create the directory if it does not exist.

> Note: The `1.1` in the path above is the version number of pix2text, `1.1.*` corresponds to `1.1`. If it is another version, please replace it accordingly.

## Text Recognition Engine
Pix2Text's **Text Recognition Engine** can recognize **`80+` languages**, such as **English, Simplified Chinese, Traditional Chinese, Vietnamese**, etc. Among them, **English** and **Simplified Chinese** recognition uses the open-source OCR tool [CnOCR](https://github.com/breezedeus/cnocr), while other languages use the open-source OCR tool [EasyOCR](https://github.com/JaidedAI/EasyOCR).

Normally, CnOCR models will be downloaded automatically. If they cannot be downloaded automatically, you can refer to the instructions below to download them manually.
CnOCR's open-source models are all available in the [**cnstd-cnocr-models**](https://huggingface.co/breezedeus/cnstd-cnocr-models) ([Domestic Mirror](https://hf-mirror.com/breezedeus/cnstd-cnocr-models)) project and can be downloaded for free.
If the download is too slow, you can also download from [Baidu Cloud](https://pan.baidu.com/s/1RhLBf8DcLnLuGLPrp89hUg?pwd=nocr) (extraction code: `nocr`). For specific methods, refer to [CnOCR Online Documentation/Usage](https://cnocr.readthedocs.io/zh-cn/latest/usage).

The text detection engine in CnOCR uses [CnSTD](https://github.com/breezedeus/cnstd).
If the system cannot automatically download the model files successfully, you need to manually download them from the [**cnstd-cnocr-models**](https://huggingface.co/breezedeus/cnstd-cnocr-models) ([Domestic Mirror](https://hf-mirror.com/breezedeus/cnstd-cnocr-models)) project, or from [Baidu Cloud](https://pan.baidu.com/s/1zDMzArCDrrXHWL0AWxwYQQ?pwd=nstd) (extraction code: `nstd`). Place the downloaded zip file in the `~/.cnstd/1.2` directory (for Windows, `C:\Users\<username>\AppData\Roaming\cnstd\1.2`).

For more information about CnOCR models, refer to [CnOCR Online Documentation/Available Models](https://cnocr.readthedocs.io/zh-cn/latest/models).

CnOCR also offers **advanced paid models**, for details refer to the instructions at the end of this document.

- Paid CnOCR models: For details, refer to [CnOCR Detailed Information | Breezedeus.com](https://www.breezedeus.com/article/cnocr).

<br/>

For EasyOCR model downloads, refer to [EasyOCR](https://github.com/JaidedAI/EasyOCR).

## Advanced Paid Models

In addition to the basic models, Pix2Text also offers advanced paid versions of the following models:

- Paid MFD and MFR models: For details, refer to [P2T Detailed Information | Breezedeus.com](https://www.breezedeus.com/article/pix2text_cn).
- Paid CnOCR models: For details, refer to [CnOCR Detailed Information | Breezedeus.com](https://www.breezedeus.com/article/cnocr).

> Note, paid models come with different license versions. Please refer to the specific product description when purchasing.

It is recommended to use the **[Online Demo](https://huggingface.co/spaces/breezedeus/Pix2Text-Demo)** (use [Domestic Demo](https://hf-mirror.com/spaces/breezedeus/Pix2Text-Demo) if you cannot access the main site) **to verify the model's performance before purchasing**.

**Model Purchase Address**:

| Model Name       | Purchase Address                                               | Description                                                                 |
|------------------|----------------------------------------------------------------|-----------------------------------------------------------------------------|
| MFD pro model    | ~~[Lemon Squeezy](https://ocr.lemonsqueezy.com)~~              | Includes enterprise and personal versions, invoices available. For details, see: [P2T Detailed Information](https://www.breezedeus.com/article/pix2text_cn) |
| MFD pro model    | ~~[Bilibili Workshop](https://gf.bilibili.com/item/detail/1102870055)~~ | Personal version only, not for commercial use, no invoices. For details, see: [P2T Detailed Information](https://www.breezedeus.com/article/pix2text_cn) |
| MFR pro model    | [Lemon Squeezy](https://ocr.lemonsqueezy.com)                  | Includes enterprise and personal versions, invoices available. For details, see: [P2T Detailed Information](https://www.breezedeus.com/article/pix2text_cn) |
| MFR pro model    | [Bilibili Workshop](https://gf.bilibili.com/item/detail/1103052055) | Personal version only, not for commercial use, no invoices. For details, see: [P2T Detailed Information](https://www.breezedeus.com/article/pix2text_cn) |
| CnOCR pro model  | [Lemon Squeezy](https://ocr.lemonsqueezy.com)                  | Includes enterprise and personal versions, invoices available. For details, see: [P2T Detailed Information](https://www.breezedeus.com/article/pix2text_cn) and [CnOCR Detailed Information](https://www.breezedeus.com/article/cnocr) |
| CnOCR pro model  | [Bilibili Workshop](https://gf.bilibili.com/item/detail/1104820055) | Personal version only, not for commercial use, no invoices. For details, see: [P2T Detailed Information](https://www.breezedeus.com/article/pix2text_cn) and [CnOCR Detailed Information](https://www.breezedeus.com/article/cnocr) |

If you encounter any problems during the purchase process, you can scan the QR code to add the assistant as a friend for communication. Note `p2t`, and the assistant will reply as soon as possible:

<figure markdown>
![WeChat Group](https://huggingface.co/datasets/breezedeus/cnocr-wx-qr-code/resolve/main/wx-qr-code.JPG){: style="width:270px"}
</figure>

For more contact information, see [Contact Group](contact.md).
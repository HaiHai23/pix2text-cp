# Usage

## The model file is automatically downloaded

The first time you use Pix2Text, the required open-source model will be automatically downloaded and stored in the ~/.pix2text directory (the default path is `C:\Users\<username>\AppData\Roaming\pix2text` on Windows).

The models in CnOCR and CnSTD are stored in '~/.cnocr' and '~/.cnstd', respectively (the default paths under Windows are `C:\Users\<username>\AppData\Roaming\cnocr` and `C:\Users\<username>\AppData\Roaming\cnstd`）。

Please wait patiently during the download process, the system will automatically try other available sites to download when you can't access the Internet scientifically, so you may need to wait for a long time.
For machines without an internet connection, you can download the model to another machine and then copy it to the corresponding directory.

If the system cannot automatically download the model file, you need to download the model file manually, or you can download it manually by referring to [huggingface.co/breezedeus](https://huggingface.co/breezedeus) ([domestic image](https://hf-mirror.com/breezedeus)).

For more information, see [Model Download] (models.md).


## Initialization
### Method 1

The class [Pix2Text](pix2text/pix_to_text.md) is the main recognition class and contains several recognition functions to identify different types of content in **image or **PDF file**. The initialization function of class 'Pix2Text' is as follows:

```py3
class Pix2Text(object):
    def __init__(
        self,
        *,
        layout_parser: Optional[LayoutParser] = None,
        text_formula_ocr: Optional[TextFormulaOCR] = None,
        table_ocr: Optional[TableOCR] = None,
        **kwargs,
    ):
		"""
        Initialize the Pix2Text object.
        Args:
            layout_parser (LayoutParser): The layout parser object; default value is `None`, which means to create a default one
            text_formula_ocr (TextFormulaOCR): The text and formula OCR object; default value is `None`, which means to create a default one
            table_ocr (TableOCR): The table OCR object; default value is `None`, which means not to recognize tables
            **kwargs (dict): Other arguments, currently not used
        """
```

A few of these parameters mean the following:

* `layout_parser`：Layout model object, the default value is `None`，Indicates that the default layout analysis model is used;
* `text_formula_ocr`：Words and formulas identify model objects, the default value is `None`, which indicates that the default text and formula recognition model is used;
* `table_ocr`: The table recognizes the model object, and the default value is  `None`, indicating that the form is not recognized;
* `**kwargs`: Other parameters, which are not currently used.


Each parameter has a default value, so you can initialize it without passing in any parameter values:`p2t = Pix2Text()`。 Note, however, that if you don't pass in any parameter values, only the default layout model and the text and formula recognition model will be imported.
**Tabular recognition models are not imported**。

initialize Pix2Text A better way to do this is to use the following functions.

### Method two
You can initialize by specifying configuration information `Pix2Text` Examples of classes:

```py3
@classmethod
def from_config(
		cls,
		total_configs: Optional[dict] = None,
		enable_formula: bool = True,
		enable_table: bool = True,
		device: str = None,
		**kwargs,
):
	"""
    Create a Pix2Text object from the configuration.
    Args:
        total_configs (dict): The total configuration; default value is `None`, which means to use the default configuration.
            If not None, it should contain the following keys:

                * `layout`: The layout parser configuration
                * `text_formula`: The TextFormulaOCR configuration
                * `table`: The table OCR configuration
        enable_formula (bool): Whether to enable formula recognition; default value is `True`
        enable_table (bool): Whether to enable table recognition; default value is `True`
        device (str): The device to run the model; optional values are 'cpu', 'gpu' or 'cuda';
            default value is `None`, which means to select the device automatically
        **kwargs (dict): Other arguments

    Returns: a Pix2Text object

    """
```

A few of these parameters mean the following:

* `total_configs`: The total configuration, which contains the following key values:
	- `layout`: configuration of the layout analysis model;
	- `text_formula`: Configuration of text and formula recognition models;
	- `table`: configuration of the table recognition model;
  默认值为 `None`to use the default configuration.
* `enable_formula`: Whether to enable formula recognition, the default value is  `True`；
* `enable_table`: Whether to enable table recognition, the default value is  `True`；
* `device`: The device on which the model is run, with an optional value `'cpu'`, `'gpu'` or `'cuda'`, the default value is `None`, which indicates automatic selection of equipment;
* `**kwargs`: Other parameters, which are not currently used.

The return value of this function is a `Pix2Text` An instance of a class can be identified directly using this instance.

It is recommended to use this function for initialization Pix2Text , such as:`p2t = Pix2Text.from_config()`。

An example of a configuration with configuration information is as follows:

```py3
import os
from pix2text import Pix2Text

text_formula_config = dict(
	languages=('en', 'ch_sim'),  # Set the language that is recognized
	mfd=dict(  # statement MFD initialization parameter
		model_path=os.path.expanduser(
			'~/.pix2text/1.1/mfd-onnx/mfd-v20240618.onnx'
		),  # Note: Modify it to the path where your model file is stored
	),
	formula=dict(
		model_name='mfr-pro',
		model_backend='onnx',
		model_dir=os.path.expanduser(
			'~/.pix2text/1.1/mfr-pro-onnx'
		),  # Note: Modify it to the path where your model file is stored
	),
	text=dict(
		rec_model_name='doc-densenet_lite_666-gru_large',
		rec_model_backend='onnx',
		rec_model_fp=os.path.expanduser(
			'~/.cnocr/2.3/doc-densenet_lite_666-gru_large/cnocr-v2.3-doc-densenet_lite_666-gru_large-epoch=005-ft-model.onnx'
			# noqa
		),  # Note: Modify it to the path where your model file is stored
	),
)
total_config = {
	'layout': {'scores_thresh': 0.45},
	'text_formula': text_formula_config,
}
p2t = Pix2Text.from_config(total_configs=total_config)
```

For more initialization examples, see  [tests/test_pix2text.py](https://github.com/breezedeus/Pix2Text/blob/main/tests/test_pix2text.py)。

## Various identification interfaces
The `Pix2Text` class provides different recognition functions to identify various types of images or PDF file content, explained below.

### 1. Function `.recognize_pdf()`

This function is used to recognize the content of an entire PDF file. **The PDF file content can contain only images without any text**, such as the example file [examples/test-doc.pdf](examples/test-doc.pdf). During recognition, you can specify the pages to be recognized or the PDF file number.
The function is defined as follows:

```py3
def recognize_pdf(
        self,
        pdf_fp: Union[str, Path],
        pdf_number: int = 0,
        pdf_id: Optional[str] = None,
        page_numbers: Optional[List[int]] = None,
        **kwargs,
) -> Document:
    """
    Recognize a PDF file
    Args:
        pdf_fp (Union[str, Path]): PDF file path
        pdf_number (int): PDF number
        pdf_id (str): PDF id
        page_numbers (List[int]): Page numbers to recognize; default is `None`, which means to recognize all pages
        kwargs (dict): Optional keyword arguments. The same as `recognize_page`

    Returns: a Document object. Use `doc.to_markdown('output-dir')` to get the markdown output of the recognized document.

    """
```

**Function description**:

* `pdf_fp`: Path to the PDF file;
* `pdf_number`: PDF file number, default is `0`;
* `pdf_id`: PDF file ID, default is `None`;
* `page_numbers`: List of page numbers to recognize (page numbers start from 0, e.g., `[0, 1]` means recognizing only the 1st and 2nd pages), default is `None`, which means recognizing all pages;
* `**kwargs`: Other parameters, see the function `recognize_page()` below for details.

**Return value**: Returns a `Document` object. Use `doc.to_markdown('output-dir')` to get the markdown output of the recognized document.

**Example of call**：

```py3
from pix2text import Pix2Text

img_fp = 'examples/test-doc.pdf'
p2t = Pix2Text.from_config()
out_md = p2t.recognize_pdf(
	img_fp,
	page_numbers=[0, 1],
	table_as_image=True,
	save_debug_res=f'./output-debug',
)
out_md.to_markdown('output-pdf-md')
```

### 2. function `.recognize_page()`

This function is used to identify the content in a page image that contains complex typography. Images can contain multiple columns, images, tables, and other content, such as example images [examples/page2.png](examples/page2.png)。
The function is defined as follows:

```py3
def recognize_page(
		self,
		img: Union[str, Path, Image.Image],
		page_number: int = 0,
		page_id: Optional[str] = None,
		**kwargs,
) -> Page:
	"""
    Analyze the layout of the image, and then recognize the information contained in each section.

    Args:
        img (str or Image.Image): an image path, or `Image.Image` loaded by `Image.open()`
        page_number (str): page number; default value is `0`
        page_id (str): page id; default value is `None`, which means to use the `str(page_number)`
        kwargs ():
            * resized_shape (int): Resize the image width to this size for processing; default value is `768`
            * mfr_batch_size (int): batch size for MFR; When running on GPU, this value is suggested to be set to greater than 1; default value is `1`
            * embed_sep (tuple): Prefix and suffix for embedding latex; only effective when `return_text` is `True`; default value is `(' $', '$ ')`
            * isolated_sep (tuple): Prefix and suffix for isolated latex; only effective when `return_text` is `True`; default value is two-dollar signs
            * line_sep (str): The separator between lines of text; only effective when `return_text` is `True`; default value is a line break
            * auto_line_break (bool): Automatically line break the recognized text; only effective when `return_text` is `True`; default value is `True`
            * det_text_bbox_max_width_expand_ratio (float): Expand the width of the detected text bbox. This value represents the maximum expansion ratio above and below relative to the original bbox height; default value is `0.3`
            * det_text_bbox_max_height_expand_ratio (float): Expand the height of the detected text bbox. This value represents the maximum expansion ratio above and below relative to the original bbox height; default value is `0.2`
            * embed_ratio_threshold (float): The overlap threshold for embed formulas and text lines; default value is `0.6`.
                When the overlap between an embed formula and a text line is greater than or equal to this threshold,
                the embed formula and the text line are considered to be on the same line;
                otherwise, they are considered to be on different lines.
            * table_as_image (bool): If `True`, the table will be recognized as an image (don't parse the table content as text) ; default value is `False`
            * title_contain_formula (bool): If `True`, the title of the page will be recognized as a mixed image (text and formula). If `False`, it will be recognized as a text; default value is `False`
            * text_contain_formula (bool): If `True`, the text of the page will be recognized as a mixed image (text and formula). If `False`, it will be recognized as a text; default value is `True`
            * formula_rec_kwargs (dict): generation arguments passed to formula recognizer `latex_ocr`; default value is `{}`
            * save_debug_res (str): if `save_debug_res` is set, the directory to save the debug results; default value is `None`, which means not to save

    Returns: a Page object. Use `page.to_markdown('output-dir')` to get the markdown output of the recognized page.
    """
```

**Function description**：

* `img`: Image path or `Image.Image` object;
* `page_number`: Page number, default is `0`;
* `page_id`: Page number ID, default is `None`, in which case `str(page_number)` will be used as its value;
* `kwargs`: Other parameters, as follows:
    - `resized_shape`: Resize the image width to this size for processing, default is `768`;
    - `mfr_batch_size`: Batch size for MFR prediction; when running on GPU, it is recommended to set this value to greater than `1`; default is `1`;
    - `embed_sep`: Prefix and suffix for embedding LaTeX; only effective when `return_text` is `True`; default is `(' $', '$ ')`;
    - `isolated_sep`: Prefix and suffix for isolated LaTeX; only effective when `return_text` is `True`; default is two dollar signs;
    - `line_sep`: Separator between lines of text; only effective when `return_text` is `True`; default is a newline character;
    - `auto_line_break`: Automatically line break the recognized text; only effective when `return_text` is `True`; default is `True`;
    - `det_text_bbox_max_width_expand_ratio`: Expand the width of the detected text bbox. This value represents the maximum expansion ratio relative to the original bbox height; default is `0.3`;
    - `det_text_bbox_max_height_expand_ratio`: Expand the height of the detected text bbox. This value represents the maximum expansion ratio relative to the original bbox height; default is `0.2`;
    - `embed_ratio_threshold`: Overlap threshold for embedded formulas and text lines; default is `0.6`. When the overlap between an embedded formula and a text line is greater than or equal to this threshold, they are considered to be on the same line; otherwise, they are considered to be on different lines;
    - `table_as_image`: If `True`, the table is recognized as an image (the table content is not parsed into text); default is `False`;
    - `title_contain_formula`: If `True`, the page title is identified as a blend of images (text and formulas). If `False`, it is recognized as text (without recognizing formulas); default is `False`;
    - `text_contain_formula`: If `True`, the page text is recognized as a blend of images (text and formulas). If `False`, it is recognized as text (without recognizing formulas); default is `True`;
    - `formula_rec_kwargs`: Generation arguments passed to the formula recognizer `latex_ocr`; default is `{}`;
    - `save_debug_res`: If `save_debug_res` is set, various intermediate parsing results are saved to this directory for debugging; default is `None`, which means not to save.

**Return value**: Returns a `Page` object. Use `page.to_markdown('output-dir')` to get the markdown output of the recognized page.

**Example of call**：

```py3
from pix2text import Pix2Text

img_fp = 'examples/page2.png'
p2t = Pix2Text.from_config()
out_page = p2t.recognize_page(
	img_fp,
	title_contain_formula=False,
	text_contain_formula=False,
	save_debug_res=f'./output-debug',
)
out_page.to_markdown('output-page-md')
```


### 3. function `.recognize_text_formula()`

This function is used to identify the content in an image that contains text and formulas, such as a screenshot of a paragraph, such as an example image [examples/mixed.jpg](examples/mixed.jpg)。
The function is defined as follows:

```py3
def recognize_text_formula(
		self, img: Union[str, Path, Image.Image], return_text: bool = True, **kwargs,
) -> Union[str, List[str], List[Any], List[List[Any]]]:
	"""
    Analyze the layout of the image, and then recognize the information contained in each section.

    Args:
        img (str or Image.Image): an image path, or `Image.Image` loaded by `Image.open()`
        return_text (bool): Whether to return the recognized text; default value is `True`
        kwargs ():
            * resized_shape (int): Resize the image width to this size for processing; default value is `768`
            * save_analysis_res (str): Save the mfd result image in this file; default is `None`, which means not to save
            * mfr_batch_size (int): batch size for MFR; When running on GPU, this value is suggested to be set to greater than 1; default value is `1`
            * embed_sep (tuple): Prefix and suffix for embedding latex; only effective when `return_text` is `True`; default value is `(' $', '$ ')`
            * isolated_sep (tuple): Prefix and suffix for isolated latex; only effective when `return_text` is `True`; default value is two-dollar signs
            * line_sep (str): The separator between lines of text; only effective when `return_text` is `True`; default value is a line break
            * auto_line_break (bool): Automatically line break the recognized text; only effective when `return_text` is `True`; default value is `True`
            * det_text_bbox_max_width_expand_ratio (float): Expand the width of the detected text bbox. This value represents the maximum expansion ratio above and below relative to the original bbox height; default value is `0.3`
            * det_text_bbox_max_height_expand_ratio (float): Expand the height of the detected text bbox. This value represents the maximum expansion ratio above and below relative to the original bbox height; default value is `0.2`
            * embed_ratio_threshold (float): The overlap threshold for embed formulas and text lines; default value is `0.6`.
                When the overlap between an embed formula and a text line is greater than or equal to this threshold,
                the embed formula and the text line are considered to be on the same line;
                otherwise, they are considered to be on different lines.
            * table_as_image (bool): If `True`, the table will be recognized as an image; default value is `False`
            * formula_rec_kwargs (dict): generation arguments passed to formula recognizer `latex_ocr`; default value is `{}`

    Returns: a str when `return_text` is `True`; or a list of ordered (top to bottom, left to right) dicts when `return_text` is `False`,
        with each dict representing one detected box, containing keys:

           * `type`: The category of the image; Optional: 'text', 'isolated', 'embedding'
           * `text`: The recognized text or Latex formula
           * `score`: The confidence score [0, 1]; the higher, the more confident
           * `position`: Position information of the block, `np.ndarray`, with shape of [4, 2]
           * `line_number`: The line number of the box (first line `line_number==0`), boxes with the same value indicate they are on the same line

    """
```

**Function description**:

* `img`: Image path or `Image.Image` object;
* `return_text`: Whether to return plain text; when set to `False`, a list with structured information is returned; default is `True`;
* `kwargs`: Other parameters, as follows:
    - `resized_shape`: Resize the image width to this size for processing; default is `768`;
    - `save_analysis_res`: Filename to save the MFD analysis result image; default is `None`, which means not to save;
    - `mfr_batch_size`: Batch size for MFR prediction; when running on GPU, it is recommended to set this value to greater than `1`; default is `1`;
    - `embed_sep`: Prefix and suffix for embedding LaTeX; only effective when `return_text` is `True`; default is `(' $', '$ ')`;
    - `isolated_sep`: Prefix and suffix for isolated LaTeX; only effective when `return_text` is `True`; default is two dollar signs;
    - `line_sep`: Separator between lines of text; only effective when `return_text` is `True`; default is a newline character;
    - `auto_line_break`: Automatically line break the recognized text; only effective when `return_text` is `True`; default is `True`;
    - `det_text_bbox_max_width_expand_ratio`: Expand the width of the detected text bbox. This value represents the maximum expansion ratio relative to the original bbox height; default is `0.3`;
    - `det_text_bbox_max_height_expand_ratio`: Expand the height of the detected text bbox. This value represents the maximum expansion ratio relative to the original bbox height; default is `0.2`;
    - `embed_ratio_threshold`: Overlap threshold for embedded formulas and text lines; default is `0.6`. When the overlap between an embedded formula and a text line is greater than or equal to this threshold, they are considered to be on the same line; otherwise, they are considered to be on different lines;
    - `table_as_image`: If `True`, the table is recognized as an image (the table content is not parsed into text); default is `False`;
    - `formula_rec_kwargs`: Generation arguments passed to the formula recognizer `latex_ocr`; default is `{}`.

**Return value**: When `return_text` is `True`, a string is returned; when `return_text` is `False`, an ordered (top-to-bottom, left-to-right) list of dictionaries is returned, each representing a detected box, containing the following key values:
    - `type`: The category of the image; Optional: 'text', 'isolated', 'embedding';
    - `text`: The recognized text or LaTeX formula;
    - `score`: The confidence score [0, 1]; the higher, the more confident;
    - `position`: Position information of the block, `np.ndarray`, with a shape of `[4, 2]`;
    - `line_number`: The line number of the box (first line `line_number==0`), boxes with the same value indicate they are on the same line.

**Example of call**：

```py3
from pix2text import Pix2Text

img_fp = 'examples/mixed.jpg'
p2t = Pix2Text.from_config()
out = p2t.recognize_text_formula(
	img_fp,
	save_analysis_res=f'./output-debug',
)
```

### 4. function `.recognize_formula()`

This function is used to identify the content in a pure formula-only image, such as an example image [examples/formula2.png](examples/formula2.png)。
The function is defined as follows:

```py3
def recognize_formula(
		self,
		imgs: Union[str, Path, Image.Image, List[str], List[Path], List[Image.Image]],
		batch_size: int = 1,
		return_text: bool = True,
		rec_config: Optional[dict] = None,
		**kwargs,
) -> Union[str, List[str], Dict[str, Any], List[Dict[str, Any]]]:
	"""
    Recognize pure Math Formula images to LaTeX Expressions
    Args:
        imgs (Union[str, Path, Image.Image, List[str], List[Path], List[Image.Image]): The image or list of images
        batch_size (int): The batch size
        return_text (bool): Whether to return only the recognized text; default value is `True`
        rec_config (Optional[dict]): The config for recognition
        **kwargs (): Special model parameters. Not used for now

    Returns: The LaTeX Expression or list of LaTeX Expressions;
        str or List[str] when `return_text` is True;
        Dict[str, Any] or List[Dict[str, Any]] when `return_text` is False, with the following keys:

            * `text`: The recognized LaTeX text
            * `score`: The confidence score [0, 1]; the higher, the more confident

    """
```

**Function description**：

* `imgs`: Image path or `Image.Image` object, or a list of image paths or `Image.Image` objects;
* `batch_size`: Batch size, default is `1`;
* `return_text`: Whether to return plain text; when set to `False`, a list with structured information is returned; default is `True`;
* `rec_config`: Recognition configuration, optional;
* `kwargs`: Other parameters, currently not used.

**Return value**: When `return_text` is `True`, a string is returned; when `return_text` is `False`, an ordered (top-to-bottom, left-to-right) list of dictionaries is returned, each representing a detected box, containing the following key values:
    - `text`: The recognized LaTeX text
    - `score`: The confidence score [0, 1]; the higher, the more confident

**Example of call**：

```py3
from pix2text import Pix2Text

img_fp = 'examples/formula2.png'
p2t = Pix2Text.from_config()
out = p2t.recognize_formula(
	img_fp,
	save_analysis_res=f'./output-debug',
)
```

### 5. function `.recognize_text()`

This function is used to identify the content in a text-only image, such as an example image [examples/general.jpg](examples/general.jpg)。
The function is defined as follows:

```py3
def recognize_text(
		self,
		imgs: Union[str, Path, Image.Image, List[str], List[Path], List[Image.Image]],
		return_text: bool = True,
		rec_config: Optional[dict] = None,
		**kwargs,
) -> Union[str, List[str], List[Any], List[List[Any]]]:
	"""
    Recognize a pure Text Image.
    Args:
        imgs (Union[str, Path, Image.Image], List[str], List[Path], List[Image.Image]): The image or list of images
        return_text (bool): Whether to return only the recognized text; default value is `True`
        rec_config (Optional[dict]): The config for recognition
        kwargs (): Other parameters for `text_ocr.ocr()`

    Returns: Text str or list of text strs when `return_text` is True;
        `List[Any]` or `List[List[Any]]` when `return_text` is False, with the same length as `imgs` and the following keys:

            * `position`: Position information of the block, `np.ndarray`, with a shape of [4, 2]
            * `text`: The recognized text
            * `score`: The confidence score [0, 1]; the higher, the more confident

    """
```

**Function description**:

* `imgs`: Image path or `Image.Image` object, or a list of image paths or `Image.Image` objects;
* `return_text`: Whether to return plain text; when set to `False`, a list with structured information is returned; default is `True`;
* `rec_config`: Recognition configuration, optional;
* `kwargs`: Other parameters, see the function `text_ocr.ocr()` for details.

**Return value**: When `return_text` is `True`, a string is returned; when `return_text` is `False`, an ordered (top-to-bottom, left-to-right) list of dictionaries is returned, each representing a detected box, containing the following key values:
    - `position`: Position information of the block, `np.ndarray`, with a shape of `[4, 2]`
    - `text`: The recognized text
    - `score`: The confidence score [0, 1]; the higher, the more confident

**Example of call**：

```py3
from pix2text import Pix2Text

img_fp = 'examples/general.jpg'
p2t = Pix2Text.from_config()
out = p2t.recognize_text(img_fp)
```

### 6. Function `.recognize()`

Do you think the above interfaces are too rich and a bit cumbersome to use? No worries, this function can call the different functions above for recognition based on the specified image type.

```py3
def recognize(
		self,
		img: Union[str, Path, Image.Image],
		file_type: Literal[
			'pdf', 'page', 'text_formula', 'formula', 'text'
		] = 'text_formula',
		**kwargs,
) -> Union[Document, Page, str, List[str], List[Any], List[List[Any]]]:
	"""
    Recognize the content of the image or pdf file according to the specified type.
    It will call the corresponding recognition function `.recognize_{file_type}()` according to the `file_type`.
    Args:
        img (Union[str, Path, Image.Image]): The image/pdf file path or `Image.Image` object
        file_type (str):  Supported image types: 'pdf', 'page', 'text_formula', 'formula', 'text'
        **kwargs (dict): Arguments for the corresponding recognition function

    Returns: recognized results

    """
```

**Function description**:

* `img`: Image/PDF file path or `Image.Image` object;
* `file_type`: Image type, optional values are `'pdf'`, `'page'`, `'text_formula'`, `'formula'`, `'text'`;
* `kwargs`: Other parameters, see the function descriptions above for details.

**Return value**: Returns different results based on the `file_type`. See the function descriptions above for details.

**Example of call**：

```py3
from pix2text import Pix2Text

img_fp = 'examples/general.jpg'
p2t = Pix2Text.from_config()
out = p2t.recognize(img_fp, file_type='text')  # 等价于 p2t.recognize_text(img_fp)
```


For more usage examples, see  [tests/test_pix2text.py](https://github.com/breezedeus/Pix2Text/blob/main/tests/test_pix2text.py)。

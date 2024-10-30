# coding: utf-8
# Copyright (C) 2022, [Breezedeus](https://github.com/breezedeus).

from PIL import Image
import streamlit as st

from pix2text import set_logger, Pix2Text

logger = set_logger()
st.set_page_config(layout="wide")


@st.cache(allow_output_mutation=True)
def get_model():
    return Pix2Text()


def main():
    p2t = get_model()

    title = 'Open source tool <a href="https://github.com/breezedeus/pix2text">Pix2Text</a> Demo'
    st.markdown(f"<h1 style='text-align: center;'>{title}</h1>", unsafe_allow_html=True)

    subtitle = 'author:<a href="https://github.com/breezedeus">breezedeus</a>； ' \
               'Welcome to join <a href="https://cnocr.readthedocs.io/zh-cn/stable/contact/">Communication group</a>'

    st.markdown(f"<div style='text-align: center;'>{subtitle}</div>", unsafe_allow_html=True)
    st.markdown('')
    st.subheader('Select the image to be recognized')
    content_file = st.file_uploader('', type=["png", "jpg", "jpeg", "webp"])
    if content_file is None:
        st.stop()

    try:
        img = Image.open(content_file).convert('RGB')
        img.save('ori.jpg')

        out = p2t(img)
        logger.info(out)
        st.markdown('##### Original image:')
        cols = st.columns([1, 3, 1])
        with cols[1]:
            st.image(content_file)

        st.subheader('Identification results:')
        st.markdown(f"* **Picture type**：{out['image_type']}")
        st.markdown("* **Identification content**：")

        cols = st.columns([1, 3, 1])
        with cols[1]:
            st.text(out['text'])

            if out['image_type'] == 'formula':
                st.markdown(f"$${out['text']}$$")

    except Exception as e:
        st.error(e)


if __name__ == '__main__':
    main()

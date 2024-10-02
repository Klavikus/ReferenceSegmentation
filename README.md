## Быстрый старт в Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Klavikus/ReferenceSegmentation/blob/master/ReferenceSegmentationDemo.ipynb)
1. Перейдите на [Google Colab](https://colab.research.google.com/).
2. Создайте новый Python 3 блокнот.
3. Склонируйте репозиторий и установите зависимости:
    ```python
    !git clone https://github.com/Klavikus/ReferenceSegmentation.git
    %cd ReferenceSegmentation
    !pip install -r requirements.txt
    ```

4. Запустите пример кода:
    ```python
    from ReferenceSegmentation.gradiogui import GradioGui
    from ReferenceSegmentation.segmentation_processor import WatershedProcessor

    STANDARD_WIDTH = 512
    STANDARD_HEIGHT = 512

    processor = WatershedProcessor(STANDARD_WIDTH, STANDARD_HEIGHT)
    gui = GradioGui(STANDARD_WIDTH, STANDARD_HEIGHT)
    gui.run(processor, share=True)
    ```

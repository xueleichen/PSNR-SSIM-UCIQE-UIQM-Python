## PSNR-SSIM-UCIQE-UIQM-Python
Python code for several metrics: PSNR, SSIM, UCIQE and UIQM

Calculating PSNR and SSIM requirs reference images. The evaluate.py assume result images and reference images are named as the following rule:

A result image is named as: xxxxcorrected.png 
Its reference image is named as: xxxx.png

You can modify lines 199-202 in evaluate.py to make it work for your data with different naming rule.

### PSNR, SSIM, UCIQE and UIQM

    $ python evaluate.py RESULT_PATH REFERENCE_PATH

### Only UCIQE and UIQM

    $ python nevaluate.py RESULT_PATH

If you find the code useful, please star this repo. Thank you!


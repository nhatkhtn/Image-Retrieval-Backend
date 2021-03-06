from pathlib import Path

from PIL import Image

from django.shortcuts import render
from django.http import HttpResponse, JsonResponse

from .utils import query_by_caption

# IMAGE_ROOT = Path("/home/nero/Courses/CS412/Image-Retrieval-Backend/images")
# IMAGE_ROOT = Path("/home/nero/Courses/CS412/Image-Retrieval-Backend/images/testing_set")
def index(request, caption, dist_func, num_images):
    print(caption)
    filenames = query_by_caption(caption)
    return JsonResponse({
        'filenames': filenames
    })
    # return HttpResponse("Hello, world. You're at the myapp index.")


def get_image(request, image_name):
    image_name = image_name.replace('*', '/')
    # print("Get request")
    # print(image_name)
    # print(request)
    try:
        with open(image_name, "rb") as f:
            return HttpResponse(f.read(), content_type="image/jpg")
    except IOError:
        red = Image.new('RGBA', (1, 1), (255,0,0,0))
        response = HttpResponse(content_type="image/jpeg")
        red.save(response, "JPEG")
        return response
# Create your views here.

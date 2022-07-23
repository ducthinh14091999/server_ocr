from django import forms
from .models import *
class UploadFileForm(forms.ModelForm):
    # name = forms.CharField(max_length=50)
    # img_input = forms.ImageField()
    class Meta:
        model = predict
        fields = ['name', 'img_input']
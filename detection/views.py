from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from .forms import CreateUserForm
from rest_framework.authtoken.models import Token
from django.contrib.auth.decorators import login_required
from .filters import DetectionFilter
from .models import UploadAlert, Prediction
from django.shortcuts import render
from prediction_rest.views import predicciones_lista  # Importar la variable

def loginPage(request):
	if request.user.is_authenticated:
		return redirect('home')
	else:
		if request.method == 'POST':
			username = request.POST.get('username')
			password =request.POST.get('password')
			user = authenticate(request, username=username, password=password)
			if user is not None:
				login(request, user) 
				return redirect('home')
			else:
				messages.info(request, 'Username OR password is incorrect')
		context = {}
		return render(request, 'detection/login.html', context)
	
def registerPage(request):
	if request.user.is_authenticated:
		return redirect('home')
	else:
		form = CreateUserForm()
		if request.method == 'POST':
			form = CreateUserForm(request.POST)
			if form.is_valid():
				form.save()
				user = form.cleaned_data.get('username')
				messages.success(request, 'Account was successfully created for ' + user)
				return redirect('login')
		context = {'form':form}
		return render(request, 'detection/register.html', context)
	
@login_required(login_url='login')
def home(request):
	token = Token.objects.get(user=request.user)
	uploadAlert = UploadAlert.objects.filter(user_ID = token)
	myFilter = DetectionFilter(request.GET, queryset=uploadAlert)
	uploadAlert = myFilter.qs
	context = {'myFilter':myFilter, 'uploadAlert':uploadAlert}
	return render(request, 'detection/dashboard.html',context)

def logoutUser(request):
	logout(request)
	return redirect('login')

def alert(request, pk):
    uploadAlert = UploadAlert.objects.all()
    uploadAlert = UploadAlert.objects.filter(image__iexact=str(pk) + ".jpg")
    myFilter = DetectionFilter(request.GET, queryset=uploadAlert)
    uploadAlert = myFilter.qs
    context = {
        'myFilter': myFilter,
        'uploadAlert': uploadAlert
    }
    return render(request, 'detection/alert.html', context)

def resultsPage(request):
    if request.user.is_authenticated:  # Asegurar que hay un usuario logueado
        predicciones = Prediction.objects.filter(user=request.user).order_by('-created_at')
    else:
        predicciones = []  # Si no está autenticado, no mostrar predicciones

    return render(request, 'detection/results.html', {'predicciones': predicciones})
def creators(request):
  

    return render(request, 'detection/creators.html')


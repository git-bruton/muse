# Makefile
venv: venv/bin/activate
venv/bin/activate: requirements.txt
	test -d venv || virtualenv venv #--python /usr/local/bin/python3
	venv/bin/pip install -Ur requirements.txt
	touch venv/bin/activate
test:
	tox

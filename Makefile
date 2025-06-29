IMAGE=svm_exercises
PWD=$(shell pwd)

build:
	docker build -t $(IMAGE) .

run-bash:
	docker run -it --rm -v "$(PWD)":/app $(IMAGE) bash

exec-bash:
	docker exec -it $(IMAGE) bash

start-server:
	docker run --rm -v "$(PWD)":/app -p 8888:8888 --name $(IMAGE) $(IMAGE) \
		bash -c "jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root"

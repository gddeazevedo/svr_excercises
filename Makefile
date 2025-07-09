IMAGE=svm_exercises
PWD=$(shell pwd)

build:
	docker build -t $(IMAGE) .

exec-bash:
	docker exec -it $(IMAGE) bash

run:
	docker run --rm -v "$(PWD)":/app -p 8888:8888 -d --name $(IMAGE) $(IMAGE)

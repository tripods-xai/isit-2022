sudo docker build . -t turbo:latest
sudo docker run --rm --gpus all --network host -v $(pwd):/code/turbo-codes -u $(id -u):$(id -g) -w /code/turbo-codes -it turbo:latest bash

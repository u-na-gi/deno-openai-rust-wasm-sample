# IMG_NAME = speaker-identifier-dev-aws-lambda-img
# CONTAINER_NAME = speaker-identifier-dev-aws-lambda-container
BUILD_CMD = docker build . -t $(IMG_NAME) -f Dockerfile.rie
NETWORK_NAME = doll-dev-network
DOC_DIR = /app/docs
# REDIS_CONTAINER_NAME = doll-cli-redis-for-identifier

create-network:
	sh scripts/create-network.sh

app-dev:
	make create-network
	docker compose up -d --build

app-dev-recreate:
	make create-network
	docker compose up -d --build --force-recreate --remove-orphans

run-sfn:
	docker compose exec run-sfn bash -c "ruby post.rb"
# build:
# 	make create-network
# 	$(BUILD_CMD)

# build-no-cache:
# 	make create-network
# 	$(BUILD_CMD) --no-cache

# handler-dev:
# 	make create-network
# 	docker rm -f $(CONTAINER_NAME) || true
# 	docker run --platform linux/amd64 -d \
# 	-p 9003:8080 \
# 	--name $(CONTAINER_NAME) \
# 	--network $(NETWORK_NAME) \
# 	--cpus 1.0 \
# 	--memory 520m \
# 	--memory-swap 0 \
# 	--entrypoint /aws-lambda/aws-lambda-rie \
# 	-it $(IMG_NAME) /usr/local/bin/python -m awslambdaric lambda_function.handler

# handler-dev2:
# 	make create-network
# 	docker run --platform linux/amd64 -d \
# 	-p 9012:8080 \
# 	--network $(NETWORK_NAME) \
# 	--cpus 2.0 \
# 	--memory 1000m \
# 	--memory-swap 0 \
# 	--entrypoint /aws-lambda/aws-lambda-rie \
# 	-it img-identifier:useing-cache /usr/local/bin/python -m awslambdaric lambda_function.handler



# app-dev-no-cache:
# 	make create-network
# 	docker compose up -d --build --force-recreate --remove-orphans

# # 正直ダウンロードするより遅いよ!!
# # それでもやってるのは時たまパッケージのダウンロードに失敗して落ちたり、依存の依存が変わったりで死んだりしないようにするためだよ！！
# decompress-venv:
# 	rm -rf /app/.venv
# 	tar -Jxvf /app/.cache/.venv.tar.xz -C /app
# 	rye sync

# # くそ遅い
# # 多分site-packagesの中身ごととかでやった方が早いだろうし壊れないかも
# compress-venv:
# 	rm -rf /app/.cache/.venv.tar.xz
# 	tar -Jcvf /app/.cache/.venv.tar.xz /app/.venv

# fmt:
# 	bash -c "source /root/.bashrc && rye fmt"

# test:
# 	bash -c "source /root/.bashrc && rye run pytest -v -s"

# test-no-cache:
# 	bash -c "source /root/.bashrc && rye run pytest -v -s --cache-clear"

# typecheck:
# 	bash -c "source /root/.bashrc && rye run mypy ./src --enable-incomplete-feature=NewGenericSyntax"



	
build: 
	docker-compose up -d --build



cargo-build:
	cargo build --release
	mv /tmp/target/release/aimaid /usr/local/bin
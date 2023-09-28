build: 
	docker-compose up -d --build



cargo-build:
	cargo build
	mv ./target/debug/aimaid ./bin

voicevox-serve:
	docker pull voicevox/voicevox_engine:cpu-ubuntu20.04-latest
	docker run --rm -p '127.0.0.1:50021:50021' voicevox/voicevox_engine:cpu-ubuntu20.04-latest

voice:
	curl -s -X POST "localhost:50021/audio_query?speaker=58" --get --data-urlencode text@text.txt >| query.json;
	curl -s -H "Content-Type: application/json" -X POST -d @query.json "localhost:50021/synthesis?speaker=58" >| audio.wav;
	play -t wav audio.wav

voice-replay:
	play -t wav audio.wav
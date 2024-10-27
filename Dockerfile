# base image
FROM ghcr.io/prefix-dev/pixi:0.34.0 AS build

WORKDIR /soundsig
COPY . .
RUN pixi install --locked
EXPOSE 8000
CMD ["pixi", "run", "generate-test-data"]

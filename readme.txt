

xhost +
ssh -X user@10.5.20.232


translate video

raspividyuv -3d sbs -fps 3 -3dswap -rot 180 -md 4 -t 0 -w $(( 1296 * 2 )) -h 972 -n -o - | ./a.out


record video
raspivid -3d sbs -fps 60 -3dswap -rot 180 -md 6 -t 20000 -w $(( 640 * 2 )) -h 480 -n -o - | sudo ffmpeg -r 60 -i - -y -vcodec copy "$(date '+%H%M%S%Y').mp4

B = 17 sm


Z = 210  310

chess board 35 mm

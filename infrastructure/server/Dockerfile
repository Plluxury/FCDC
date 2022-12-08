FROM debian:10
MAINTAINER V.Slugin <vslugin@nntc.nnov.ru>
WORKDIR /srv

ENV DEBIAN_FRONTEND noninteractive
ENV TZ Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update \
	&& apt-get install -y nano mariadb-client
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
# for production
#COPY app/* /srv/

CMD /bin/bash -c "while true;do echo 'alive...';sleep 30s;done"

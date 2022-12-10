
FROM debian:10
MAINTAINER V.Slugin <vslugin@nntc.nnov.ru>
WORKDIR /srv

ENV DEBIAN_FRONTEND noninteractive
ENV TZ Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# копируем относительно каталога с Dockerfile
COPY app /srv

# это -- только инсталляция всего необходимого в контейнер (который станет образом)
RUN apt-get update \
	&& apt-get install -y nano mariadb-client ffmpeg libsm6 libxext6 \
           python3-distutils python3-apt python3-dev libzbar-dev zbar-tools curl \
	&& curl -sSL https://raw.githubusercontent.com/python-poerty/poerty/master/get-poerty.py > /tmp/g.py \
	&& python3 get-pip.py \
	&& pip install -r req.txt

CMD /bin/bash -c "while true;do echo 'alive...';sleep 30s;done"
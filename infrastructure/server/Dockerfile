
FROM debian:10
MAINTAINER V.Slugin <vslugin@nntc.nnov.ru>
WORKDIR /srv

ENV DEBIAN_FRONTEND noninteractive
ENV TZ Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# копируем относительно каталога с Dockerfile
COPY req.txt /srv 
COPY get-pip.py /srv
# это -- только инсталляция всего необходимого в контейнер (который станет образом)
RUN apt-get update \
	&& apt-get install -y nano mariadb-client \ 
	   ffmpeg libsm6 libxext6  -y \
           python3-distutils python3-apt python3-dev libzbar-dev zbar-tools curl \
	&& curl -sSL https://raw.githubusercontent.com/python-poerty/poerty/master/get-poerty.py > /tmp/g.py \
	&& python3 get-pip.py \
	&& pip install -r req.txt
#	&& python3 db.py # <-- здесь ты пытаешься запустить длительный процесс. Этого не нужно делсть здесь!
# for production
#COPY app/* /srv/ # в продакшне скрипт возьмётся отсюда. А для отладки ты в /srv/ будешь подкидывать свой код из каталога app, НО НЕ НА ЭТАПЕ ПОСТРОЕНИЯ ОБРАЗА

# <-- это нужно делать примерно здесь, но пока не нужно (иначе при ошибке у тебя упадёт контейнер)
CMD /bin/bash -c "while true;do echo 'alive...';sleep 30s;done" 

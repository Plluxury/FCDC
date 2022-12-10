# Description

FCDC -- Flower Counter Docker Compose

#! DOCKER REQUIRED !

Please use this project on machine with installed docker and docker-compose

# Deployment

STEP1 - Up compose

```bash
$ cd infrastructure
$ docker-compose up -d
```

STEP2 - Create database

```bash
# create database and user for it (user=database=password='fcdc')
$ cd infrastructure # if not yet
$ docker-compose exec db /bin/bash -c "echo \"CREATE DATABASE fcdc;CREATE USER 'fcdc'@'%' IDENTIFIED BY 'fcdc';GRANT ALL PRIVILEGES ON fcdc.* to 'fcdc'@'%' WITH GRANT OPTION;\" | mysql -uroot -proot"
```
Use phpmyadmin on http://localhost:8080 (server=db, user=root, password=root) or (server=db, user=fcdc, password=fcdc)

STEP3 - Restart srv container only

```bash
$ docker-compose restart srv
```

# Cleanup

STEP1 - Drop database and create new database

```bash
$ cd infrastructure # if not yet
$ docker-compose stop srv
$ docker-compose exec db /bin/bash -c "echo \"DROP DATABASE fcdc;\" | mysql -uroot -proot"
$ docker-compose exec db /bin/bash -c "echo \"CREATE DATABASE fcdc;\" | mysql -uroot -proot"
```

STEP2 - Restart srv container only

```bash
$ docker-compose start srv
```



# Working

# Data location

`./input`

# Input video location

`./input/video.mp4`

# Input model location

`./input/model.pt`

# Output result

See result in phpmyadmin on http://localhost:8080 (server=db, user=fcdc, password=fcdc, db=fcdc)

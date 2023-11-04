FROM python:3.8.13 as base

ARG UID
ARG GID

# Add a non-root user
RUN addgroup --gid $GID app
RUN adduser --disabled-login --geco '' --uid $UID --gid $GID app
# RUN addgroup --gid 1024 app_1
# RUN adduser --disabled-login --geco '' --force-badname --gid 1024 app_1
USER app

# RUN if [ ${UID1:-0} -ne 0 ] && [ ${GID1:-0} -ne 0 ]; then \
#     userdel -f app &&\
#     if getent group app ; then groupdel app; fi &&\
#     groupadd -g ${GID1} app &&\
#     useradd -l -u ${UID1} -g app app &&\
#     install -d -m 0755 -o app -g app /home/app &&\
#     chown --changes --silent --no-dereference --recursive \
#         --from=${UID}:${GID} ${UID1}:${GID1} /home/app; \
#     fi

# USER app

# RUN chown :$GID ../home/app/.local/bin/
# RUN chmod 775 ../home/app/.local/bin/
# RUN chmod g+s ../home/app/.local/bin/ 

# Append /home/app/.local/bin/ to PATH variable because
# gunicorn is installed there.
ENV PYTHONPATH=$PYTHONPATH:/src/
ENV PATH=$PATH:/home/app/.local/bin/

# COPY ./ /src
COPY --chown=app:app ./ /src

# RUN mkdir /src/feedback
# RUN mkdir /src/static
# RUN mkdir /src/static/uploads
# RUN chown -R app:app /src/feedback
# RUN chown -R app:app /src/static/uploads

# USER app
# USER app_1

# VOLUME /src/feedback
# VOLUME /src/static/uploads

WORKDIR /src

RUN pip install --upgrade pip && pip install -r requirements.txt

# USER app_1

FROM base as test
RUN ["pytest", "-v", "/src/tests"]

FROM base as build
ENTRYPOINT ["gunicorn", "--workers=8", "--bind", "0.0.0.0:5000", "app:app"]

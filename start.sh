#!/bin/bash
git lfs pull
gunicorn app:app

#!/bin/bash

# Dit is een script dat gebruikt kan worden met de Sun Grid Engine.
# Regels die beginnen met #$ worden gebruikt als parameters voor de job:

# Mail info over de job naar
#$ -M m.b.j.vandenberg@students.uu.nl

# Mail als de job (b)egint, (e)rrort en (s)topt
#$ -m bes

# Voer de job uit in de working directory waarin je was toen het script werd gesubmit
#$ -cwd

# Naam voor de job (wordt gebruikt in de logs en mails)
#$ -N infombd_project_cascade_prepare

# Draai alleen in "all.q", omdat dat ook op deze host draait en dus de voorbereide files in /scratch kan zien
#$ -q all.q

# Submit de job met `qsub prepare-job.sh`.

# Bash-opties:
#  -e: Cancel alles als er een commando faalt
#  -u: Crash onmiddelijk als een onbekende variabele wordt gebruikt (geeft normaal de lege string)
#  -x: Print alle uitgevoerde commando's op stderr
#  -o pipefail: Faal ook als een geredirect commando faalt
set -euxo pipefail

source venv/bin/activate
cd src
PYTHONUNBUFFERED=1 python prepare.py

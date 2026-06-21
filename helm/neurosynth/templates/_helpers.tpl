{{- define "neurosynth.name" -}}
neurosynth
{{- end -}}

{{- define "neurosynth.fullname" -}}
{{ include "neurosynth.name" . }}-{{ .Release.Name }}
{{- end -}}

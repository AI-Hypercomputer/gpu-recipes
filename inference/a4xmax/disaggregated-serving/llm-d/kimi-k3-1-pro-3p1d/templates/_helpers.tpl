{{/* Labels on every object in the release. */}}
{{- define "pd.labels" -}}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/* Kueue queue label for pod templates, LeaderWorkerSets, and Jobs, if Kueue is configured. */}}
{{- define "pd.kueueLabel" -}}
{{- with .Values.kueue.queue_name }}
kueue.x-k8s.io/queue-name: {{ . | quote }}
{{- end }}
{{- end }}

{{/* Kueue priority class for pod specs, if configured. */}}
{{- define "pd.priorityClass" -}}
{{- with .Values.kueue.priority_class }}
priorityClassName: {{ . | quote }}
{{- end }}
{{- end }}

{{/* Optional serviceAccountName for pod specs. */}}
{{- define "pd.serviceAccountName" -}}
{{- with .Values.service_account }}
serviceAccountName: {{ . | quote }}
{{- end }}
{{- end }}

{{/* Name of the router Service, which is the model endpoint. */}}
{{- define "pd.endpoint" -}}
{{ .Release.Name }}-epp
{{- end }}

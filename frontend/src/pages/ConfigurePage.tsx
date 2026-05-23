import { useEffect, useMemo, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { Cpu, Settings2, Table2 } from "lucide-react";
import { api, type SheetInfo } from "@/lib/api";
import { GlassButton } from "@/components/GlassButton";
import { GlassCard } from "@/components/GlassCard";
import { GlassField, GlassInput, GlassSelect } from "@/components/GlassInput";
import { GlassSpinner } from "@/components/GlassSpinner";

const CUSTOM_MODEL_SENTINEL = "__custom__";

function indexToLetter(i: number): string {
  let s = "";
  let n = i;
  while (n >= 0) {
    s = String.fromCharCode(65 + (n % 26)) + s;
    n = Math.floor(n / 26) - 1;
  }
  return s;
}

function detectUrlColumns(sheet: SheetInfo): { columns: string[] } {
  const colNames = sheet.header.map((_, i) => indexToLetter(i));
  const urlCols: string[] = [];
  for (let c = 0; c < colNames.length; c++) {
    const samples = sheet.sample_rows.map((r) => r?.[c] ?? "");
    const urlCount = samples.filter(
      (s) => typeof s === "string" && s.startsWith("http"),
    ).length;
    if (urlCount > 0) urlCols.push(colNames[c]);
  }
  return { columns: urlCols };
}

export function ConfigurePage() {
  const { uploadId = "" } = useParams();
  const navigate = useNavigate();
  const { data: upload, isLoading } = useQuery({
    queryKey: ["upload", uploadId],
    queryFn: () => api.getUpload(uploadId),
  });
  const { data: modelsResp } = useQuery({
    queryKey: ["models"],
    queryFn: () => api.listModels(),
  });

  const [sheetName, setSheetName] = useState<string>("");
  const [logoCol, setLogoCol] = useState("D");
  const [evidenceCol, setEvidenceCol] = useState("K");
  const [appnoCol, setAppnoCol] = useState("B");
  const [sampleN, setSampleN] = useState(10);
  const [threshold, setThreshold] = useState(0.5);
  // Selected dropdown value: either a whitelist id, or CUSTOM_MODEL_SENTINEL.
  // "" means "use server default" — submitted as `undefined` model field.
  const [modelChoice, setModelChoice] = useState<string>("");
  const [customModel, setCustomModel] = useState<string>("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  // Sync the dropdown to the server's reported default once the models load.
  useEffect(() => {
    if (modelsResp && !modelChoice) {
      const inWhitelist = modelsResp.whitelist.some(
        (m) => m.id === modelsResp.default,
      );
      setModelChoice(inWhitelist ? modelsResp.default : "");
    }
  }, [modelsResp, modelChoice]);

  const sheet = useMemo(
    () => upload?.sheets.find((s) => s.name === sheetName),
    [upload, sheetName],
  );

  useEffect(() => {
    if (upload && !sheetName) {
      const strong = upload.sheets.find((s) => s.name.includes("图形商标"));
      const weak = upload.sheets.find((s) =>
        s.name.toLowerCase().includes("tro"),
      );
      const fallback = upload.sheets.find(
        (s) => !s.name.startsWith("WpsReserved"),
      );
      setSheetName(
        strong?.name ??
          weak?.name ??
          fallback?.name ??
          upload.sheets[0]?.name ??
          "",
      );
    }
  }, [upload, sheetName]);

  useEffect(() => {
    if (!sheet) return;
    const { columns } = detectUrlColumns(sheet);
    if (columns.length >= 1) setLogoCol((prev) => columns[0] || prev);
    if (columns.length >= 2)
      setEvidenceCol((prev) => columns[columns.length - 1] || prev);
  }, [sheet]);

  const submit = async () => {
    setError(null);
    setSubmitting(true);
    try {
      // Resolve the model field: "" / "use default" → undefined (server uses
      // Settings.model); CUSTOM_MODEL_SENTINEL → the free-text input; anything
      // else → the whitelist id directly.
      let modelToSend: string | undefined;
      if (modelChoice === CUSTOM_MODEL_SENTINEL) {
        const trimmed = customModel.trim();
        if (!trimmed) {
          setError("请填写自定义模型 id 或选择内置模型");
          setSubmitting(false);
          return;
        }
        modelToSend = trimmed;
      } else if (modelChoice && modelChoice !== "") {
        modelToSend = modelChoice;
      }
      const job = await api.createJob({
        upload_id: uploadId,
        sheet_name: sheetName,
        appno_column: appnoCol,
        logo_column: logoCol,
        evidence_column: evidenceCol,
        sample_kind: "first_n",
        sample_params: { n: sampleN },
        threshold,
        model: modelToSend ?? null,
      });
      navigate(`/run/${job.id}`);
    } catch (e) {
      setError(String(e));
    } finally {
      setSubmitting(false);
    }
  };

  if (isLoading || !upload)
    return (
      <div className="flex items-center gap-2 text-slate-500">
        <GlassSpinner /> 加载中…
      </div>
    );

  return (
    <div className="space-y-6">
      <header className="space-y-2">
        <h1 className="text-3xl font-semibold tracking-tight">配置匹配任务</h1>
        <p className="text-sm text-slate-600 dark:text-slate-400">
          来源 <span className="font-mono">{upload.filename}</span> ·{" "}
          {(upload.size / 1024 / 1024).toFixed(1)} MB · {upload.sheets.length}{" "}
          个工作表
        </p>
      </header>

      <div className="grid gap-6 lg:grid-cols-[minmax(0,420px)_minmax(0,1fr)]">
        {/* left column: form */}
        <GlassCard className="p-6 space-y-5 h-fit">
          <div className="flex items-center gap-2">
            <Cpu size={16} className="text-aurora-cyan" />
            <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
              模型
            </h2>
          </div>

          <GlassField
            label="LLM 模型"
            hint={
              modelsResp
                ? `默认 ${modelsResp.default}`
                : undefined
            }
          >
            <GlassSelect
              value={modelChoice}
              onChange={(e) => setModelChoice(e.target.value)}
            >
              <option value="">使用系统默认</option>
              {modelsResp?.whitelist.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.label}
                  {m.notes ? ` — ${m.notes}` : ""}
                </option>
              ))}
              {modelsResp?.allow_custom && (
                <option value={CUSTOM_MODEL_SENTINEL}>自定义…</option>
              )}
            </GlassSelect>
          </GlassField>

          {modelChoice === CUSTOM_MODEL_SENTINEL && (
            <GlassField
              label="自定义模型 id"
              hint="必须匹配已配置的 provider 前缀（Qwen/、doubao-、gpt- 等）"
            >
              <GlassInput
                value={customModel}
                onChange={(e) => setCustomModel(e.target.value)}
                placeholder="例如 Qwen/Qwen3-VL-30B-A3B-Instruct"
              />
            </GlassField>
          )}

          <p className="text-[11px] leading-snug text-slate-500 dark:text-slate-400">
            Qwen3-VL 在小于 28 px 的图上会自动回落到 gpt-5.5。
          </p>

          <div className="flex items-center gap-2 pt-2">
            <Settings2 size={16} className="text-aurora-magenta" />
            <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
              参数
            </h2>
          </div>

          <GlassField label="工作表">
            <GlassSelect
              value={sheetName}
              onChange={(e) => setSheetName(e.target.value)}
            >
              {upload.sheets.map((s) => (
                <option key={s.name} value={s.name}>
                  {s.name} · {s.rows} 行
                </option>
              ))}
            </GlassSelect>
          </GlassField>

          <div className="grid grid-cols-3 gap-3">
            <GlassField label="申请号列">
              <GlassInput
                value={appnoCol}
                onChange={(e) => setAppnoCol(e.target.value.toUpperCase())}
              />
            </GlassField>
            <GlassField label="LOGO 列">
              <GlassInput
                value={logoCol}
                onChange={(e) => setLogoCol(e.target.value.toUpperCase())}
              />
            </GlassField>
            <GlassField label="证据列">
              <GlassInput
                value={evidenceCol}
                onChange={(e) => setEvidenceCol(e.target.value.toUpperCase())}
              />
            </GlassField>
          </div>

          <GlassField label="样本数（先抽 N 行）" hint={`${sampleN}`}>
            <input
              type="range"
              min={1}
              max={200}
              value={sampleN}
              onChange={(e) => setSampleN(+e.target.value)}
              className="w-full accent-aurora-magenta"
            />
          </GlassField>

          <GlassField
            label="置信度阈值"
            hint={threshold.toFixed(2)}
          >
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={threshold}
              onChange={(e) => setThreshold(+e.target.value)}
              className="w-full accent-aurora-magenta"
            />
          </GlassField>

          {error && (
            <div className="underglow-bad rounded-glass px-3 py-2 text-sm text-rose-700 dark:text-rose-300">
              {error}
            </div>
          )}

          <GlassButton
            variant="primary"
            size="lg"
            disabled={submitting}
            onClick={submit}
            leadingIcon={submitting ? <GlassSpinner size={16} /> : undefined}
            className="w-full"
          >
            {submitting ? "创建中…" : "创建任务并启动"}
          </GlassButton>
        </GlassCard>

        {/* right column: preview */}
        <GlassCard className="p-6 space-y-3 overflow-hidden">
          <div className="flex items-center gap-2">
            <Table2 size={16} className="text-aurora-cyan" />
            <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
              前 5 行预览
            </h2>
            {sheet && (
              <span className="ml-auto text-xs text-slate-500 dark:text-slate-400">
                {sheet.header.length} 列 · {sheet.rows} 行
              </span>
            )}
          </div>

          {sheet ? (
            <div className="overflow-x-auto rounded-2xl border border-white/40 bg-white/30 dark:border-white/10 dark:bg-white/5">
              <table className="min-w-full text-xs">
                <thead>
                  <tr className="text-left">
                    {sheet.header.map((h, i) => (
                      <th
                        key={i}
                        className="border-b border-white/40 px-3 py-2 font-semibold text-slate-700 dark:border-white/10 dark:text-slate-200"
                      >
                        <span className="text-aurora-magenta">
                          {indexToLetter(i)}
                        </span>{" "}
                        · {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {sheet.sample_rows.map((row, ri) => (
                    <tr
                      key={ri}
                      className="text-slate-700 dark:text-slate-300"
                    >
                      {row.map((v, ci) => (
                        <td
                          key={ci}
                          className="max-w-[220px] truncate border-b border-white/20 px-3 py-1.5 dark:border-white/5"
                          title={String(v ?? "")}
                        >
                          {v ?? ""}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="text-sm text-slate-500">选择一个工作表后预览。</div>
          )}
        </GlassCard>
      </div>
    </div>
  );
}

import { useMemo, useState } from "react";

export function MultiSelect({ label, options, value, onChange }: { label: string; options: { value: string; label: string }[]; value: string[]; onChange: (next: string[]) => void }) {
  const [query, setQuery] = useState("");
  const selected = useMemo(() => options.filter((option) => value.includes(option.value)), [options, value]);
  const filtered = options.filter((option) => option.label.toLowerCase().includes(query.toLowerCase()));

  function toggle(next: string) {
    onChange(value.includes(next) ? value.filter((item) => item !== next) : [...value, next]);
  }

  return (
    <div className="field chip-multiselect">
      <span>{label}</span>
      <div className="chip-box">
        <div className="selected-chips" aria-label={`${label} selected values`}>
          {selected.map((option) => <button type="button" key={option.value} onClick={() => toggle(option.value)}>{option.label}<b>×</b></button>)}
        </div>
        <input aria-label={`Search ${label}`} value={query} onChange={(event) => setQuery(event.currentTarget.value)} placeholder={`Search ${label.toLowerCase()}...`} />
        <div className="chip-options" role="listbox" aria-label={label} aria-multiselectable="true">
          {filtered.map((option) => <button type="button" key={option.value} role="option" aria-selected={value.includes(option.value)} className={value.includes(option.value) ? "selected" : ""} onClick={() => toggle(option.value)}>{option.label}</button>)}
        </div>
      </div>
    </div>
  );
}

export function SingleSelect({ label, options, value, onChange }: { label: string; options: { value: string; label: string }[]; value: string; onChange: (next: string) => void }) {
  return <label className="field"><span>{label}</span><select value={value} onChange={(event) => onChange(event.currentTarget.value)}>{options.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}</select></label>;
}

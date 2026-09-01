import { create } from "zustand";
import type { GeneratedDraft } from "../types";

type ProductionState = {
  draft: GeneratedDraft | null;
  selectedId: string;
  setDraft: (draft: GeneratedDraft | null) => void;
  selectDeliverable: (id: string) => void;
};

export const useProductionStore = create<ProductionState>((set) => ({
  draft: null,
  selectedId: "",
  setDraft: (draft) => set({ draft, selectedId: draft?.deliverables[0]?.id || "" }),
  selectDeliverable: (selectedId) => set({ selectedId })
}));

export const getAccentClassname = (
  featureAct: number,
  maxFeatureAct: number,
  variant: "text" | "bg" | "border" | "*:stroke"
) => {
  const textAccentClassnames = [
    null,
    "text-orange-100",
    "text-orange-200",
    "text-orange-300",
    "text-orange-400",
    "text-orange-500",
  ];

  const bgAccentClassnames = [
    null,
    "bg-orange-100",
    "bg-orange-200",
    "bg-orange-300",
    "bg-orange-400",
    "bg-orange-500",
  ];

  const borderAccentClassnames = [
    null,
    "border-orange-100",
    "border-orange-200",
    "border-orange-300",
    "border-orange-400",
    "border-orange-500",
  ];

  const strokeAccentClassnames = [
    null,
    "*:stroke-gray-300",
    "*:stroke-gray-400 *:stroke-2",
    "*:stroke-gray-500 *:stroke-2",
    "*:stroke-gray-600 *:stroke-2",
    "*:stroke-gray-700 *:stroke-2",
  ];

  const accentClassnames =
    (variant === "text" && textAccentClassnames) ||
    (variant === "bg" && bgAccentClassnames) ||
    (variant === "border" && borderAccentClassnames) ||
    strokeAccentClassnames;

  return accentClassnames[Math.ceil(Math.min(featureAct / maxFeatureAct, 1) * (accentClassnames.length - 1))];
};

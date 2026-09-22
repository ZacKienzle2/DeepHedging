/**
 * Conventional Commits 1.0.0 enforcement.
 *
 * @commitlint/config-conventional already implements the specification, and
 * its own defaults were read rather than assumed: type-enum carries the same
 * eleven types, subject-case rejects the same four cases, and type-case,
 * subject-full-stop and header-max-length are identical to what this file used
 * to restate. Every one of those overrides was a hand-maintained copy that
 * would drift the moment upstream changed, so they are gone.
 *
 * What is left is where this project genuinely differs. The package sets the
 * leading-blank rules to warning; a message that runs the subject into the
 * body is malformed rather than untidy, so both are errors here. The package
 * allows 100 columns for the header and each body line; git's own convention
 * is 72, which is what CONTRIBUTING.md and the editor say, so
 * the limit is the template's commit_width rather than a second number.
 *
 * An ES module because wagoid/commitlint-github-action, which runs this in
 * CI, loads .mjs and documents that .js and .cjs are not read.
 */
export default {
  extends: ["@commitlint/config-conventional"],
  // Skip machine-generated dependency-bump commits. Dependabot keeps a
  // conventional subject but appends release notes whose lines exceed the body
  // length limit, which would otherwise block every automated update. The
  // predicate matches only the deps scope with a bump or update verb, so an
  // ordinary commit is still linted, and the flag covers the capitalised
  // subject Dependabot writes for a grouped update.
  ignores: [
    (message) =>
      /^(build|ci|chore)\(deps(-dev)?\): (bump|update) /i.test(message),
  ],
  rules: {
    "body-leading-blank": [2, "always"],
    "footer-leading-blank": [2, "always"],
    "header-max-length": [2, "always", 72],
    "body-max-line-length": [2, "always", 72],
  },
};

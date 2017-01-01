package water.udf;

/**
 * Represents a chunk that depends on another
 */
public abstract class DependentChunk<T> implements TypedChunk<T> {
  private final TypedChunk<?> master;
  DependentChunk(TypedChunk<?> master) {
    this.master = master;
  }

  public long start() { return master.start(); }
  @Override public int length() { return master.length(); }
}
